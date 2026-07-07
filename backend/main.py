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
import traceback
from importlib import import_module
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from huggingface_hub import HfApi

# --- IMPORTS ---
from src.utils.hf_hub import (
    search_hf_models,
    get_model_access_issue,
    is_model_access_blocked,
    build_model_access_error,
    build_hf_load_error,
)
from src.utils.hf_auth import hf_auth_kwargs
from src.wrappers.hf_text_classification import HFTextClassificationWrapper
from src.wrappers.hf_text_generation import HFTextGenerationWrapper
from src.wrappers.hf_image import HFImageWrapper
from src.wrappers.hf_image_classification import HFImageClassificationWrapper
from src.utils.image_attribution import decode_base64_image
from src.attributors.captum_grad import CaptumGradientsAttributor
from src.attributors.captum_deeplift import CaptumDeepLiftAttributor
from src.attributors.captum_saliency import CaptumSaliencyAttributor
from src.attributors.captum_inputxgrad import CaptumInputXGradientAttributor
from src.attributors.captum_gradientshap import CaptumGradientShapAttributor
from src.attributors.captum_occlusion import CaptumOcclusionAttributor
from src.attributors.captum_lime import CaptumLimeAttributor

from src.db import create_job, update_job_success, update_job_failed, get_job, get_all_jobs, delete_job, delete_all_jobs, set_job_pinned

# Global Mutex to prevent concurrent threads from crashing the GPU during inference
gpu_lock = threading.Lock()

# --- SETUP PATHS ---
BASE_DIR = Path(__file__).resolve().parent
HF_CACHE_DIR = BASE_DIR / "hf_models_cache"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
print(f"hf cache dir set to: {HF_CACHE_DIR}")
os.environ["HF_HOME"] = str(HF_CACHE_DIR)

DEFAULT_DEVICE_ENV_VAR = "LUMIXAI_DEFAULT_DEVICE"
UNRECOVERABLE_CUDA_ERROR_MARKERS = (
    "device-side assert triggered",
)
UNRECOVERABLE_CUDA_RESTART_MESSAGE = (
    "Unrecoverable CUDA error detected. "
    "The backend is restarting automatically, so please retry in a few seconds."
)

_backend_restart_lock = threading.Lock()
_backend_restart_scheduled = False

def normalize_requested_device(requested_device: Optional[str] = "auto") -> str:
    device = (requested_device or "auto").strip().lower()
    if device == "auto":
        device = os.getenv(DEFAULT_DEVICE_ENV_VAR, "auto").strip().lower() or "auto"
    return device

def is_unrecoverable_cuda_error(error: Any) -> bool:
    error_text = str(error).lower()
    return any(marker in error_text for marker in UNRECOVERABLE_CUDA_ERROR_MARKERS)

def schedule_backend_restart(reason: str) -> None:
    global _backend_restart_scheduled

    with _backend_restart_lock:
        if _backend_restart_scheduled:
            return
        _backend_restart_scheduled = True

    print(
        "Scheduling backend restart after unrecoverable CUDA error: "
        f"{reason}"
    )

    def restart_backend_process() -> None:
        time.sleep(1.0)
        os._exit(1)

    threading.Thread(target=restart_backend_process, daemon=True).start()

def build_cuda_restart_error(detail: str) -> str:
    return f"{UNRECOVERABLE_CUDA_RESTART_MESSAGE} Original error: {detail}"

def is_cuda_device(device: Optional[str]) -> bool:
    return bool(device) and device.startswith("cuda")

def is_indexed_cuda_device(device: Optional[str]) -> bool:
    return bool(device) and device.startswith("cuda:")

def clear_cuda_memory(device: Optional[str] = None) -> None:
    if not torch.cuda.is_available():
        return

    target_device = device if is_indexed_cuda_device(device) else None
    if target_device:
        with torch.cuda.device(target_device):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        return

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def get_optimal_device(requested_device: str = "auto") -> str:
    """Determines the best available hardware accelerator.

    Args:
        requested_device (str, optional): The user's preference ("auto", "cpu", "cuda",
            "cuda:0", "cuda:1", "mps"). Defaults to "auto".

    Returns:
        str: The optimal device string compatible with PyTorch.
    """
    resolved_request = normalize_requested_device(requested_device)

    if resolved_request == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if resolved_request == "cpu":
        return "cpu"

    if resolved_request == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested, but no NVIDIA GPU is available to the backend.")
        return "cuda"

    if resolved_request.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ValueError(f"{resolved_request} was requested, but no NVIDIA GPU is available to the backend.")

        try:
            gpu_index = int(resolved_request.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device '{resolved_request}'. Use 'cuda' or 'cuda:<index>'.") from exc

        visible_gpu_count = torch.cuda.device_count()
        if gpu_index < 0 or gpu_index >= visible_gpu_count:
            raise ValueError(
                f"Requested CUDA device '{resolved_request}' is not available. "
                f"The backend can currently see {visible_gpu_count} GPU(s)."
            )
        return resolved_request

    if resolved_request == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS was requested, but it is not available on this machine.")
        return "mps"

    raise ValueError(
        f"Unsupported device '{resolved_request}'. "
        "Use one of: auto, cpu, cuda, cuda:<index>, mps."
    )

# --- 1. REGISTRY ---
AVAILABLE_WRAPPERS = {
    "hf_text_classification": HFTextClassificationWrapper,
    "hf_text_generation": HFTextGenerationWrapper,
    "hf_image": HFImageWrapper,
    "hf_image_classification": HFImageClassificationWrapper,
}

AVAILABLE_SOURCES = [
    {"id": "huggingface", "name": "Hugging Face Hub", "type": "remote"},
]

# Wrapper names an attributor can be applied to, keyed on the same strings produced by
# the task-detection `match` in /api/load. Used both to validate /api/set_attributor
# requests server-side and to let the frontend gray out incompatible options.
TEXT_WRAPPERS = ["hf_text_classification", "hf_text_generation"]
IMAGE_CLASSIFICATION_WRAPPERS = ["hf_image_classification"]
# The 7 Captum attributors operate on continuous tensors (token embeddings for text,
# pixel values for images) and dispatch internally on wrapper type, so they're
# compatible with both modalities. DAAM is diffusion-specific and stays image-generation only.
CAPTUM_COMPATIBLE_WRAPPERS = TEXT_WRAPPERS + IMAGE_CLASSIFICATION_WRAPPERS

AVAILABLE_ATTRIBUTORS = {
    "captum_ig": {
        "name": "Integrated Gradients (Captum)",
        "class": CaptumGradientsAttributor,
        "compatible_wrappers": CAPTUM_COMPATIBLE_WRAPPERS,
    },
    "captum_deeplift": {
        "name": "DeepLift (Captum)",
        "class": CaptumDeepLiftAttributor,
        "compatible_wrappers": CAPTUM_COMPATIBLE_WRAPPERS,
    },
    "captum_saliency": {
        "name": "Saliency (Captum)",
        "class": CaptumSaliencyAttributor,
        "compatible_wrappers": CAPTUM_COMPATIBLE_WRAPPERS,
    },
    "captum_inputxgrad": {
        "name": "Input x Gradient (Captum)",
        "class": CaptumInputXGradientAttributor,
        "compatible_wrappers": CAPTUM_COMPATIBLE_WRAPPERS,
    },
    "captum_gradientshap": {
        "name": "GradientSHAP (Captum)",
        "class": CaptumGradientShapAttributor,
        "compatible_wrappers": CAPTUM_COMPATIBLE_WRAPPERS,
    },
    "captum_occlusion": {
        "name": "Occlusion (Captum)",
        "class": CaptumOcclusionAttributor,
        "compatible_wrappers": CAPTUM_COMPATIBLE_WRAPPERS,
    },
    "captum_lime": {
        "name": "LIME (Captum)",
        "class": CaptumLimeAttributor,
        "compatible_wrappers": CAPTUM_COMPATIBLE_WRAPPERS,
    },
    "daam": {
        "name": "DAAM (Diffusion Attentive Attribution Maps)",
        "import_path": "src.attributors.daam:DAAMAttributor",
        "compatible_wrappers": ["hf_image"],
    },
}

MODEL_SEARCH_LIMIT = 25


def resolve_attributor_class(attributor_id: str):
    attributor_config = AVAILABLE_ATTRIBUTORS[attributor_id]

    if "class" in attributor_config:
        return attributor_config["class"]

    module_path, class_name = attributor_config["import_path"].split(":", 1)
    module = import_module(module_path)
    return getattr(module, class_name)


def get_available_attributors():
    available_attributors = []

    for attributor_id, metadata in AVAILABLE_ATTRIBUTORS.items():
        try:
            resolve_attributor_class(attributor_id)
            available_attributors.append({
                "id": attributor_id,
                "name": metadata["name"],
                "compatible_wrappers": metadata.get("compatible_wrappers", []),
            })
        except Exception as exc:
            print(f"Skipping unavailable attributor '{attributor_id}': {exc}")

    return available_attributors

# --- 2. GLOBAL STATE ---
app_state: Dict[str, Any] = {
    "active_wrapper": None,
    "active_wrapper_name": None,
    "active_attributor": None,
    "active_attributor_id": None,
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
    device: str = "auto"

class AttributorRequest(BaseModel):
    attributor_id: str
    params: Optional[Dict[str, Any]] = {}

class ExplainRequest(BaseModel):
    text: Optional[str] = None
    # Base64-encoded input image, used instead of `text` for image classification models.
    image_base64: Optional[str] = None
    target_class: Optional[int] = None
    ignore_special_tokens: bool = True
    seed: Optional[int] = None
    max_new_tokens: Optional[int] = None
    disable_thinking: bool = False
    # DAAM image-generation overrides. When None the attributor keeps its own
    # defaults. negative_prompt == "" explicitly disables the negative prompt.
    guidance_scale: Optional[float] = None
    negative_prompt: Optional[str] = None

class JobResponse(BaseModel):
    job_id: str
    status: str

class PinRequest(BaseModel):
    pinned: bool

# --- 4. SETUP APP ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 5. BACKGROUND ---
def run_explanation_task(job_id: str, text: Optional[str], target_class: Optional[int], ignore_special_tokens: bool = True, seed: Optional[int] = None, guidance_scale: Optional[float] = None, negative_prompt: Optional[str] = None, disable_thinking: bool = False, max_new_tokens: Optional[int] = None, image_base64: Optional[str] = None):
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
        seed (Optional[int], optional): Seed for reproducible generation. Currently honored by
            attributors that involve stochastic generation (e.g. DAAM). Ignored otherwise.
        disable_thinking (bool, optional): Requests non-thinking mode for supported
            text-generation chat templates. Ignored by non-text-generation tasks.
        max_new_tokens (Optional[int], optional): Text-generation token budget. If None,
            the wrapper uses the configured backend default.
    """
    with gpu_lock:
        # Start the clock only after acquiring the GPU lock, so execution_time_sec
        # reflects the actual compute time and not the time spent queued behind
        # other jobs waiting for the lock.
        start_time = time.time()
        try:
            attributor = app_state["active_attributor"]
            wrapper = app_state["active_wrapper"]

            if not attributor or not wrapper:
                raise ValueError("Modello o Attributor disconnessi durante l'esecuzione")

            input_data = decode_base64_image(image_base64) if image_base64 else text

            output = attributor.attribute(
                input_data=input_data,
                target_output=target_class,
                ignore_special_tokens=ignore_special_tokens,
                seed=seed,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                disable_thinking=disable_thinking,
                max_new_tokens=max_new_tokens
            )
            predicted_word = None

            if output.target == "text_generation":
                payload = {
                    "target_id": "text_generation",
                    "predicted_token": None,
                    "tokens": [],
                    "scores": output.heatmap,
                    "generated_image": None,
                    # Tokenizer metadata (attribution values untouched) so the frontend can
                    # optionally hide special and/or chat-template tokens from the visualization.
                    "input_special_mask": output.metadata.get("input_special_mask"),
                    "output_special_mask": output.metadata.get("output_special_mask"),
                    "input_template_mask": output.metadata.get("input_template_mask"),
                }
            else:
                if isinstance(output.target, int):
                    if hasattr(wrapper, "get_predicted_label"):
                        try:
                            predicted_word = wrapper.get_predicted_label(output.target)
                        except:
                            pass
                    elif hasattr(wrapper, "tokenizer"):
                        try:
                            predicted_word = wrapper.tokenizer.decode([output.target])
                        except:
                            pass

                payload = {
                    "target_id": output.target,
                    "predicted_token": predicted_word,
                    "tokens": [f.content for f in output.input_features],
                    "scores": output.heatmap.tolist() if hasattr(output.heatmap, "tolist") else output.heatmap,
                    "generated_image": output.generated_image,
                    # Tokenizer metadata (attribution values untouched); None for image tasks,
                    # where DAAM already filters special tokens during generation.
                    "special_tokens_mask": output.metadata.get("special_tokens_mask"),
                    # Original uploaded image (base64), set only by image classification
                    # attributors; lets the frontend distinguish this from text classification.
                    "input_image": output.metadata.get("input_image"),
            }

            end_time = time.time()
            was_persisted = update_job_success(job_id, payload, start_time, end_time)
            if not was_persisted:
                print(f"Job '{job_id}' was deleted before completion. Discarding result payload.")

        except Exception as e:
            traceback.print_exc()
            error_message = str(e)
            was_updated = update_job_failed(job_id, error_message)
            if not was_updated:
                print(f"Job '{job_id}' was deleted before failure could be persisted.")

            if is_unrecoverable_cuda_error(error_message):
                schedule_backend_restart(error_message)


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
        "attributors": get_available_attributors()
    }

@app.get("/api/search", response_model=List[SearchResult])
def search_models(source: str, q: str, limit: int = MODEL_SEARCH_LIMIT):
    """Proxies the search request to the appropriate external Hub."""
    if source == "huggingface":
        bounded_limit = max(1, min(limit, 50))
        return search_hf_models(query=q, limit=bounded_limit)
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
            previous_device = getattr(app_state["active_wrapper"], "device", None)
            print("Cleaning up memory and VRAM...")
            del app_state["active_wrapper"]
            del app_state["active_attributor"]
            app_state["active_wrapper"] = None
            app_state["active_wrapper_name"] = None
            app_state["active_attributor"] = None
            app_state["active_attributor_id"] = None
            gc.collect()
            clear_cuda_memory(previous_device)
        
        real_device = get_optimal_device(req.device)
        if is_indexed_cuda_device(real_device):
            torch.cuda.set_device(torch.device(real_device))

        wrapper_instance = None
        wrapper_name = "unknown"
        detected_task = "unknown"

        if req.source == "huggingface":            
            api = HfApi(**hf_auth_kwargs())
            try:
                info = api.model_info(req.model_name)
            except Exception as e:
                error_message = build_hf_load_error(req.model_name, e)
                if error_message != str(e):
                    raise HTTPException(400, error_message)
                info = None

            if info is not None:
                access_issue = get_model_access_issue(info)
                if is_model_access_blocked(access_issue):
                    raise HTTPException(400, build_model_access_error(req.model_name, access_issue))
                detected_task = info.pipeline_tag or "-"
            else:
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
                case "image-classification":
                    wrapper_instance = HFImageClassificationWrapper(req.model_name, real_device)
                    wrapper_name = "hf_image_classification"
                case _:
                    wrapper_instance = HFTextClassificationWrapper(req.model_name, real_device)
                    wrapper_name = "hf_text_classification (fallback)"
        else:
            raise HTTPException(400, f"Source '{req.source}' not supported")

        # Strip any " (fallback)" suffix so this matches the plain wrapper-name keys
        # used in AVAILABLE_ATTRIBUTORS[...]["compatible_wrappers"].
        app_state["active_wrapper"] = wrapper_instance
        app_state["active_wrapper_name"] = wrapper_name.split(" ")[0]
        app_state["active_attributor"] = None
        app_state["active_attributor_id"] = None
        app_state["active_source"] = req.source
        app_state["active_model_name"] = req.model_name
        
        return {
            "status": "loaded", "model": req.model_name,
            "wrapper": wrapper_name, "device": real_device,
            "detected_task": detected_task
        }

    except ValueError as e:
        raise HTTPException(400, str(e))
    except HTTPException as e:
        if is_unrecoverable_cuda_error(e.detail):
            schedule_backend_restart(str(e.detail))
            raise HTTPException(e.status_code, build_cuda_restart_error(str(e.detail)))
        raise
    except Exception as e:
        error_message = build_hf_load_error(req.model_name, e)
        if is_unrecoverable_cuda_error(error_message):
            schedule_backend_restart(error_message)
            raise HTTPException(503, build_cuda_restart_error(error_message))
        status_code = 400 if error_message != str(e) else 500
        raise HTTPException(status_code, error_message)

@app.post("/api/unload")
def unload_model():
    """Releases VRAM and clears the global state. Protected by the GPU Mutex lock."""
    with gpu_lock:
        try:
            if app_state.get("active_wrapper") is not None:
                previous_device = getattr(app_state["active_wrapper"], "device", None)
                del app_state["active_wrapper"]
                del app_state["active_attributor"]
                app_state["active_wrapper"] = None
                app_state["active_wrapper_name"] = None
                app_state["active_attributor"] = None
                app_state["active_attributor_id"] = None
                app_state["active_source"] = None
                app_state["active_model_name"] = None
                
                gc.collect()
                clear_cuda_memory(previous_device)
                    
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

    compatible_wrappers = AVAILABLE_ATTRIBUTORS[req.attributor_id].get("compatible_wrappers", [])
    active_wrapper_name = app_state.get("active_wrapper_name")
    if compatible_wrappers and active_wrapper_name not in compatible_wrappers:
        attributor_name = AVAILABLE_ATTRIBUTORS[req.attributor_id]["name"]
        raise HTTPException(
            400,
            f"Attributor '{attributor_name}' is not compatible with the loaded model "
            f"(detected type: '{active_wrapper_name}'). It requires one of: {', '.join(compatible_wrappers)}.",
        )

    try:
        AttrClass = resolve_attributor_class(req.attributor_id)
        app_state["active_attributor"] = AttrClass(app_state["active_wrapper"])
        app_state["active_attributor_id"] = req.attributor_id
        return {"status": "active", "id": req.attributor_id, "name": AVAILABLE_ATTRIBUTORS[req.attributor_id]["name"]}
    except Exception as e:
        raise HTTPException(400, f"Attributor '{req.attributor_id}' is not currently available: {e}")


@app.post("/api/explain", response_model=JobResponse)
def explain(req: ExplainRequest, background_tasks: BackgroundTasks):
    """Enqueues a new asynchronous XAI job."""
    if not app_state.get("active_attributor") or not app_state.get("active_wrapper"):
        raise HTTPException(400, "Model and attributor must be loaded first.")
    
    source_name = app_state.get("active_source", "unknown")
    model_name = app_state.get("active_model_name", "unknown")
    active_attributor_id = app_state.get("active_attributor_id")
    attributor_name = AVAILABLE_ATTRIBUTORS.get(active_attributor_id, {}).get("name", "Unknown")

    if req.max_new_tokens is not None and req.max_new_tokens <= 0:
        raise HTTPException(400, "max_new_tokens must be a positive integer.")

    if not req.text and not req.image_base64:
        raise HTTPException(400, "Either 'text' or 'image_base64' must be provided.")

    job_prompt = req.text if req.text else "[Uploaded Image]"
    job_id = create_job(job_prompt, source_name, model_name, attributor_name)

    background_tasks.add_task(run_explanation_task, job_id, req.text, req.target_class, req.ignore_special_tokens, req.seed, req.guidance_scale, req.negative_prompt, req.disable_thinking, req.max_new_tokens, req.image_base64)

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

@app.patch("/api/jobs/{job_id}/pin")
def pin_job(job_id: str, req: PinRequest):
    """Pins or unpins a job so it can be kept at the top of the history list."""
    was_updated = set_job_pinned(job_id, req.pinned)
    if not was_updated:
        raise HTTPException(404, "Job not found")
    return {"status": "success", "pinned": req.pinned}

@app.delete("/api/jobs/{job_id}")
def delete_job_by_id(job_id: str):
    """Deletes a single job record and its persisted payload, if present."""
    try:
        was_deleted = delete_job(job_id)
        if not was_deleted:
            raise HTTPException(404, "Job not found")
        return {"status": "success", "message": "Job deleted successfully."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error occurred while deleting the job: {str(e)}")

@app.delete("/api/jobs")
def clear_all_jobs():
    """Deletes all job records from the SQLite DB and removes JSON payloads from the disk."""
    try:
        delete_all_jobs()
        return {"status": "success", "message": "Database and result files cleared."}
    except Exception as e:
        raise HTTPException(500, f"Error occurred while clearing the database: {str(e)}")
