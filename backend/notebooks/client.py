import requests
import time
from typing import List, Dict, Any
from collections import defaultdict
from tqdm.auto import tqdm


class Client:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self._check_connection()

    def _check_connection(self):
        try:
            res = requests.get(f"{self.base_url}/")
            res.raise_for_status()
            print(f"Backend Online: {res.json().get('status')}")
        except Exception:
            print(f"Error connecting to {self.base_url}.")

    def _load_model(self, source: str, model_name: str, device: str = "auto"):
        res = requests.post(f"{self.base_url}/api/load", json={"source": source, "model_name": model_name, "device": device})
        res.raise_for_status()
        return res.json()

    def _set_attributor(self, attributor_id: str):
        res = requests.post(f"{self.base_url}/api/set_attributor", json={"attributor_id": attributor_id})
        res.raise_for_status()
        return res.json()

    def run_smart_batch(self, jobs: List[Dict[str, Any]], poll_interval: float = 2.0, sort_strategy: str = "fastest_first") -> List[Dict[str, Any]]:
        """
        Run a batch of jobs with smart grouping and optional sorting.
        sort_strategy can be: "fastest_first", "slowest_first", or "none".
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

        with tqdm(total=total_jobs, desc="Elaboration") as pbar:
            
            # 2. SEQUENTIAL GROUP PROCESSING
            for (source, model_name, attributor_id), group in sorted_groups:
                print(f"\nConfiguration for model: '{model_name}' | Attributor: '{attributor_id}'...")
                
                # Load model (once per group) and set attributor for the current group
                self._load_model(source, model_name)
                self._set_attributor(attributor_id)
                
                job_ids_in_flight = []
                
                # Submit all jobs in the current group
                for job in group:
                    res = requests.post(f"{self.base_url}/api/explain", json={
                        "text": job['prompt'],
                        "target_class": job.get('target_class', None),
                        "ignore_special_tokens": job.get('ignore_special_tokens', False)
                    })
                    res.raise_for_status()
                    job_ids_in_flight.append((job['_original_index'], res.json()["job_id"]))

                # 3. GROUP POLLING
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
                        
        print("\n Global batch completed!")
        return results