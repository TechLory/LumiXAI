"""FastAPI Backend Entrypoint for LumiXAI.

This module exposes the REST API consumed by the frontend and external clients.
It manages the global application state (loaded models and attributors), handles 
asynchronous background tasks for heavy AI inference, and guarantees thread-safe 
GPU access using a global Mutex lock.
"""

import os
import time
import torch
import gc
import threading
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from huggingface_hub import HfApi

# --- IMPORTS ---
from src.utils.hf_hub import search_hf_models
from src.wrappers.hf_text_classification import HFTextClassificationWrapper
from src.wrappers.hf_text_generation import HFTextGenerationWrapper
from src.wrappers.hf_image import HFImageWrapper
from src.attributors.daam import DAAMAttributor
from src.attributors.captum_grad import CaptumGradientsAttributor

from src.db import create_job, update_job_success, update_job_failed, get_job, get_all_jobs, delete_all_jobs

# Global Mutex to prevent concurrent threads from crashing the GPU during inference
gpu_lock = threading.Lock()

# --- SETUP PATHS ---
BASE_DIR = Path(__file__).resolve().parent
HF_CACHE_DIR = BASE_DIR / "hf_models_cache"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
print(f"hf cache dir set to: {HF_CACHE_DIR}")
os.environ["HF_HOME"] = str(HF_CACHE_DIR)

def get_optimal_device(requested_device: str = "auto") -> str:
    """Determines the best available hardware accelerator.

    Args:
        requested_device (str, optional): The user's preference ("auto", "cpu", "cuda", "mps"). Defaults to "auto".

    Returns:
        str: The optimal device string compatible with PyTorch.
    """
    if requested_device == "cpu":
        return "cpu"
    if torch.cuda.is_available() and requested_device in ["auto", "cuda"]:
        return "cuda"
    if torch.backends.mps.is_available() and requested_device in ["auto", "cuda", "mps"]:
        return "mps"
    return "cpu"

# --- 1. REGISTRY ---
AVAILABLE_WRAPPERS = {
    "hf_text_classification": HFTextClassificationWrapper,
    "hf_text_generation": HFTextGenerationWrapper,
    "hf_image": HFImageWrapper,
}

AVAILABLE_SOURCES = [
    {"id": "huggingface", "name": "Hugging Face Hub", "type": "remote"},
]

AVAILABLE_ATTRIBUTORS = {
    "captum_ig": {"name": "Integrated Gradients (Captum)", "class": CaptumGradientsAttributor},
    "daam": {"name": "DAAM (Diffusion Attentive Attribution Maps)", "class": DAAMAttributor},
}

# --- 2. GLOBAL STATE ---
app_state: Dict[str, Any] = {
    "active_wrapper": None,
    "active_attributor": None,
    "active_source": None,
    "active_model_name": None
}

# --- 3. DATA MODELS (Pydantic) ---
class SearchResult(BaseModel):
    id: str
    task: str
    likes: int
    downloads: int

class LoadRequest(BaseModel):
    source: str
    model_name: str
    device: str = "cpu"

class AttributorRequest(BaseModel):
    attributor_id: str
    params: Optional[Dict[str, Any]] = {}

class ExplainRequest(BaseModel):
    text: str
    target_class: Optional[int] = None
    ignore_special_tokens: bool = True

class JobResponse(BaseModel):
    job_id: str
    status: str

# --- 4. SETUP APP ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 5. BACKGROUND ---
def run_explanation_task(job_id: str, text: str, target_class: Optional[int], ignore_special_tokens: bool = True):
    """Executes the XAI attribution logic asynchronously.

    This function runs in a separate thread. It acquires the global `gpu_lock` to ensure 
    that only one heavy inference process hits the GPU at a time, preventing Out-Of-Memory 
    errors or DAAM tracing conflicts. Upon completion, it formats the payload and updates 
    the SQLite database.

    Args:
        job_id (str): The UUID of the database job to update.
        text (str): The input prompt provided by the user.
        target_class (Optional[int]): The specific class to attribute towards (if applicable).
        ignore_special_tokens (bool, optional): Whether to filter out structural tokens. Defaults to True.
    """
    start_time = time.time()
    with gpu_lock:
        try:
            attributor = app_state["active_attributor"]
            wrapper = app_state["active_wrapper"]

            if not attributor or not wrapper:
                raise ValueError("Modello o Attributor disconnessi durante l'esecuzione")

            output = attributor.attribute(
                input_data=text, 
                target_output=target_class, 
                ignore_special_tokens=ignore_special_tokens
            )
            predicted_word = None

            if output.target == "text_generation":
                payload = {
                    "target_id": "text_generation",
                    "predicted_token": None,
                    "tokens": [], 
                    "scores": output.heatmap,
                    "generated_image": None
                }
            else:
                if hasattr(wrapper, "tokenizer") and isinstance(output.target, int):
                    try:
                        predicted_word = wrapper.tokenizer.decode([output.target])
                    except:
                        pass

                payload = {
                    "target_id": output.target,
                    "predicted_token": predicted_word,
                    "tokens": [f.content for f in output.input_features],
                    "scores": output.heatmap.tolist() if hasattr(output.heatmap, "tolist") else output.heatmap,
                    "generated_image": output.generated_image
            }

            end_time = time.time()
            update_job_success(job_id, payload, start_time, end_time)

        except Exception as e:
            import traceback
            traceback.print_exc()
            update_job_failed(job_id, str(e))


# --- 6. ENDPOINTS ---
@app.get("/")
def health_check():
    """Returns the basic health status of the API."""
    return {"status": "LumiXAI Backend Online", "version": "0.2.0"}

@app.get("/api/manifest")
def get_manifest():
    """Returns the list of available model sources and attributor algorithms."""
    return {
        "sources": AVAILABLE_SOURCES,
        "attributors": [{"id": k, "name": v["name"]} for k, v in AVAILABLE_ATTRIBUTORS.items()]
    }

@app.get("/api/search", response_model=List[SearchResult])
def search_models(source: str, q: str):
    """Proxies the search request to the appropriate external Hub."""
    if source == "huggingface":
        return search_hf_models(query=q, limit=10)
    return []

@app.post("/api/load")
def load_model(req: LoadRequest):
    """Loads a model into the global application state.
    
    This endpoint automatically identifies the task type (e.g., text-generation vs image-generation) 
    via the Hugging Face API and instantiates the correct Wrapper class. It also aggressively 
    cleans the VRAM before loading to prevent memory overflows.
    """
    try:
        if app_state.get("active_wrapper") is not None:
            print("Cleaning up memory and VRAM...")
            del app_state["active_wrapper"]
            del app_state["active_attributor"]
            app_state["active_wrapper"] = None
            app_state["active_attributor"] = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        
        real_device = get_optimal_device(req.device)
        wrapper_instance = None
        wrapper_name = "unknown"
        detected_task = "unknown"

        if req.source == "huggingface":            
            api = HfApi()
            try:
                info = api.model_info(req.model_name)
                detected_task = info.pipeline_tag
            except Exception:
                detected_task = "-" 

            match detected_task:
                case "text-classification" | "fill-mask" | "token-classification":
                    wrapper_instance = HFTextClassificationWrapper(req.model_name, real_device)
                    wrapper_name = "hf_text_classification"
                case "text-generation" | "text2text-generation" | "translation" | "summarization":
                    wrapper_instance = HFTextGenerationWrapper(req.model_name, real_device) 
                    wrapper_name = "hf_text_generation"
                case "text-to-image":
                    wrapper_instance = HFImageWrapper(req.model_name, real_device)
                    wrapper_name = "hf_image"
                case _:
                    wrapper_instance = HFTextClassificationWrapper(req.model_name, real_device)
                    wrapper_name = "hf_text_classification (fallback)"
        else:
            raise HTTPException(400, f"Source '{req.source}' not supported")

        app_state["active_wrapper"] = wrapper_instance
        app_state["active_attributor"] = None
        app_state["active_source"] = req.source
        app_state["active_model_name"] = req.model_name
        
        return {
            "status": "loaded", "model": req.model_name,
            "wrapper": wrapper_name, "device": real_device,
            "detected_task": detected_task
        }

    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/unload")
def unload_model():
    """Releases VRAM and clears the global state. Protected by the GPU Mutex lock."""
    with gpu_lock:
        try:
            if app_state.get("active_wrapper") is not None:
                del app_state["active_wrapper"]
                del app_state["active_attributor"]
                app_state["active_wrapper"] = None
                app_state["active_attributor"] = None
                app_state["active_source"] = None
                app_state["active_model_name"] = None
                
                gc.collect()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    
                return {"status": "success", "message": "Cleaned up memory and VRAM. Model unloaded."}
            else:
                return {"status": "success", "message": "No model in memory to unload."}
        except Exception as e:
            raise HTTPException(500, f"Error occurred while cleaning up memory: {str(e)}")

@app.post("/api/set_attributor")
def set_attributor(req: AttributorRequest):
    """Instantiates and attaches an Attributor algorithm to the currently loaded model."""
    if not app_state["active_wrapper"]:
        raise HTTPException(400, "No model loaded.")
    if req.attributor_id not in AVAILABLE_ATTRIBUTORS:
        raise HTTPException(400, "Attributor ID not found")
    try:
        AttrClass = AVAILABLE_ATTRIBUTORS[req.attributor_id]["class"]
        app_state["active_attributor"] = AttrClass(app_state["active_wrapper"])
        return {"status": "active", "id": req.attributor_id, "name": AVAILABLE_ATTRIBUTORS[req.attributor_id]["name"]}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/explain", response_model=JobResponse)
def explain(req: ExplainRequest, background_tasks: BackgroundTasks):
    """Enqueues a new asynchronous XAI job."""
    if not app_state.get("active_attributor") or not app_state.get("active_wrapper"):
        raise HTTPException(400, "Model and attributor must be loaded first.")
    
    source_name = app_state.get("active_source", "unknown")
    model_name = app_state.get("active_model_name", "unknown")
    
    attr_class = type(app_state["active_attributor"])
    attributor_name = next((v["name"] for v in AVAILABLE_ATTRIBUTORS.values() if v["class"] == attr_class), "Unknown")

    job_id = create_job(req.text, source_name, model_name, attributor_name)
    
    background_tasks.add_task(run_explanation_task, job_id, req.text, req.target_class, req.ignore_special_tokens)
    
    return {"job_id": job_id, "status": "running"}

@app.get("/api/jobs")
def get_jobs():
    """Retrieves metadata for all historical jobs."""
    return get_all_jobs()

@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    """Retrieves the status and the payload (if completed) of a specific job."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job

@app.delete("/api/jobs")
def clear_all_jobs():
    """Deletes all job records from the SQLite DB and removes JSON payloads from the disk."""
    try:
        delete_all_jobs()
        return {"status": "success", "message": "Database and result files cleared."}
    except Exception as e:
        raise HTTPException(500, f"Error occurred while clearing the database: {str(e)}")