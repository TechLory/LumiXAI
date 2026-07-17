import requests
import time
import uuid
from typing import List, Dict, Any
from collections import defaultdict
from tqdm import tqdm


class Client:
    """Batch client for the LumiXAI backend.

    Each instance is a session: it claims the model it loads, so a browser user cannot
    replace the model halfway through a batch, and it passes the backend's configuration
    token with every job, so a job that would run against someone else's model fails
    loudly instead of returning an attribution of the wrong thing.

    Args:
        base_url (str): Backend root URL.
        force (bool): Take the model over even when another session still holds it.
            Off by default, so a batch cannot trample someone's live session by accident.
    """

    def __init__(self, base_url: str = "http://localhost:8000", force: bool = False):
        self.base_url = base_url.rstrip("/")
        self.session_id = uuid.uuid4().hex
        self.force = force
        self.config_id = None
        self._check_connection()

    @property
    def _headers(self) -> Dict[str, str]:
        return {"X-LumiXAI-Session": self.session_id}

    def _check_connection(self):
        try:
            res = requests.get(f"{self.base_url}/")
            res.raise_for_status()
            print(f"Backend Online: {res.json().get('status')}")
        except Exception:
            print(f"Error connecting to {self.base_url}.")

    def _load_model(self, source: str, model_name: str, device: str = "auto"):
        res = requests.post(
            f"{self.base_url}/api/load",
            json={"source": source, "model_name": model_name, "device": device, "force": self.force},
            headers=self._headers,
        )
        if res.status_code == 423:
            raise RuntimeError(
                f"{res.json().get('detail')} Pass force=True to Client(...) to take it over."
            )
        res.raise_for_status()
        data = res.json()
        self.config_id = data.get("config_id")
        return data

    def _set_attributor(self, attributor_id: str):
        res = requests.post(
            f"{self.base_url}/api/set_attributor",
            json={"attributor_id": attributor_id, "force": self.force},
            headers=self._headers,
        )
        res.raise_for_status()
        data = res.json()
        self.config_id = data.get("config_id")
        return data

    def clear_history(self) -> Dict[str, Any]:
        """
        Clear all jobs from the database and delete associated JSON result files.
        """
        print("Requesting database and file cleanup...")
        res = requests.delete(f"{self.base_url}/api/jobs")
        res.raise_for_status()
        data = res.json()
        print(f"{data.get('message')}")
        return data

    def free_memory(self) -> Dict[str, Any]:
        """
        Request the backend to free VRAM by unloading the current model and clearing caches.
        """
        print("Requesting VRAM cleanup...")
        res = requests.post(f"{self.base_url}/api/unload", json={"force": self.force}, headers=self._headers)
        res.raise_for_status()
        data = res.json()
        print(f"{data.get('message')}")
        return data

    def run_smart_batch(self, jobs: List[Dict[str, Any]], poll_interval: float = 2.0, sort_strategy: str = "fastest_first", device: str = "auto") -> List[Dict[str, Any]]:
        """
        Run a batch of jobs with smart grouping and optional sorting.
        sort_strategy can be: "fastest_first", "slowest_first", or "none".
        device selects where models are loaded ("auto", "cuda:1", ...).
        Each job may carry optional generation controls such as "seed",
        "max_new_tokens", and "disable_thinking".
        """
        if not jobs:
            return []

        # 1. JOB GROUPING
        grouped_jobs = defaultdict(list)
        for i, job in enumerate(jobs):
            job['_original_index'] = i 
            key = (job['source'], job['model'], job['attributor'])
            grouped_jobs[key].append(job)

        results = [None] * len(jobs)
        total_jobs = len(jobs)
        
        # --- SORTING LOGIC (STRATEGY PATTERN) ---
        def approx_weight(key):
            attributor = key[2]
            if attributor == "captum_ig": return 1  # Text (Fast)
            if attributor == "daam": return 2       # Images (Slow)
            return 3                                # Other
        
        if sort_strategy == "fastest_first":
            sorted_groups = sorted(grouped_jobs.items(), key=lambda item: approx_weight(item[0]))
            strategy_description = "Shortest Job First (Text -> Images)"
        elif sort_strategy == "slowest_first":
            sorted_groups = sorted(grouped_jobs.items(), key=lambda item: approx_weight(item[0]), reverse=True)
            strategy_description = "Longest Job First (Images -> Text)"
        else:
            sorted_groups = list(grouped_jobs.items())
            strategy_description = "None (Original Order)"

        print(f"\nStarting Smart Batch: {total_jobs} jobs in {len(grouped_jobs)} groups.")
        print(f"Strategy: {strategy_description}")

        pbar = tqdm(total=total_jobs, desc="Elaboration")
        try:
            # 2. SEQUENTIAL GROUP PROCESSING
            for (source, model_name, attributor_id), group in sorted_groups:
                print(f"\nConfiguration for model: '{model_name}' | Attributor: '{attributor_id}'...")
                
                self._load_model(source, model_name, device)
                self._set_attributor(attributor_id)
                
                job_ids_in_flight = []
                
                for job in group:
                    print(f"\nRequesting explanation for: '{job['prompt'][:50]}...'")
                    res = requests.post(f"{self.base_url}/api/explain", json={
                        "text": job['prompt'],
                        "target_class": job.get('target_class', None),
                        "ignore_special_tokens": job.get('ignore_special_tokens', False),
                        "seed": job.get('seed', None),
                        "guidance_scale": job.get('guidance_scale', None),
                        "negative_prompt": job.get('negative_prompt', None),
                        "max_new_tokens": job.get('max_new_tokens', None),
                        "disable_thinking": job.get('disable_thinking', False),
                        "use_chat_template": job.get('use_chat_template', True),
                        # Ties the job to the model this group loaded: if it is gone, the
                        # backend refuses rather than explaining a different model.
                        "config_id": self.config_id,
                    }, headers=self._headers)
                    res.raise_for_status()
                    job_ids_in_flight.append((job['_original_index'], res.json()["job_id"]))

                pending = job_ids_in_flight.copy()
                while pending:
                    still_pending = []
                    for original_idx, job_id in pending:
                        res = requests.get(f"{self.base_url}/api/jobs/{job_id}")
                        res.raise_for_status()
                        job_data = res.json()
                        
                        if job_data["status"] in ["completed", "failed"]:
                            results[original_idx] = job_data
                            pbar.update(1)
                        else:
                            still_pending.append((original_idx, job_id))
                            
                    pending = still_pending
                    if pending:
                        time.sleep(poll_interval)
        finally:
            pbar.close()

        print("\n Global batch completed!")
        return results