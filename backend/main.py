import os
import time
import torch
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

# --- SETUP PATHS ---
BASE_DIR = Path(__file__).resolve().parent
HF_CACHE_DIR = BASE_DIR / "hf_models_cache"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
print(f"hf cache dir set to: {HF_CACHE_DIR}")
os.environ["HF_HOME"] = str(HF_CACHE_DIR)

def get_optimal_device(requested_device: str = "auto") -> str:
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
    # allow_origins=["http://localhost:3000"],
    allow_origins=["*"], # Allow all origins for development; restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 5. BACKGROUND ---
def run_explanation_task(job_id: str, text: str, target_class: Optional[int], ignore_special_tokens: bool = True):
    """Funzione che esegue l'XAI pesante in background."""
    start_time = time.time()
    try:
        attributor = app_state["active_attributor"]
        wrapper = app_state["active_wrapper"]
        
        if not attributor or not wrapper:
            raise ValueError("Modello o Attributor disconnessi durante l'esecuzione")

        # Esecuzione
        output = attributor.attribute(
            input_data=text, 
            target_output=target_class, 
            ignore_special_tokens=ignore_special_tokens
        )
        predicted_word = None
        
        # Formattazione Output
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
        # Save in DB
        update_job_success(job_id, payload, start_time, end_time)

    except Exception as e:
        import traceback
        traceback.print_exc()
        update_job_failed(job_id, str(e))


# --- 6. ENDPOINTS ---
@app.get("/")
def health_check():
    return {"status": "XAI Backend Online", "version": "0.2.0"}

@app.get("/api/manifest")
def get_manifest():
    return {
        "sources": AVAILABLE_SOURCES,
        "attributors": [{"id": k, "name": v["name"]} for k, v in AVAILABLE_ATTRIBUTORS.items()]
    }

@app.get("/api/search", response_model=List[SearchResult])
def search_models(source: str, q: str):
    if source == "huggingface":
        return search_hf_models(query=q, limit=10)
    return []

@app.post("/api/load")
def load_model(req: LoadRequest):
    try:
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
                    wrapper_instance = HFTextGenerationWrapper(req.model_name, real_device) # pyright: ignore
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

@app.post("/api/set_attributor")
def set_attributor(req: AttributorRequest):
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


# --- NEW ENDPOINTS ---

@app.post("/api/explain", response_model=JobResponse)
def explain(req: ExplainRequest, background_tasks: BackgroundTasks):
    """Create the job and add it to the background queue."""
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
    """Retrieve all the job for the side bar"""
    return get_all_jobs()

@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    """Retrieve the status of a specific job and its payload if completed."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job

@app.delete("/api/jobs")
def clear_all_jobs():
    """Delete all jobs and their associated result files."""
    try:
        delete_all_jobs()
        return {"status": "success", "message": "Database and result files cleared."}
    except Exception as e:
        raise HTTPException(500, f"Error occurred while clearing the database: {str(e)}")