"""FastAPI Backend Entrypoint for LumiXAI.

This module exposes the REST API consumed by the frontend and external clients.
It manages the global application state (loaded models and attributors), handles
asynchronous background tasks for heavy AI inference, and guarantees thread-safe
GPU access using a global Mutex lock.

A loaded model stays resident in (V)RAM until it is replaced or explicitly unloaded,
so a background reaper thread releases it once it has gone unused for
`LUMIXAI_MODEL_IDLE_TIMEOUT_SEC` seconds. See the "IDLE MODEL LIFECYCLE" section.

The backend holds exactly one model at a time, but several clients may talk to it at
once. Two mechanisms keep that honest, both in the "SESSION OWNERSHIP" section:

* a **config token** (`config_id`), minted on every load and attributor change. Clients
  echo it back on /api/explain, and a job whose token no longer matches the live
  configuration is rejected instead of being silently computed on someone else's model.
* a **session lease**, so a client that loaded a model keeps the right to it until the
  model falls idle. Others are told who holds it and must take over deliberately.
"""

import os
import time
import torch
import gc
import threading
import traceback
import uuid
from contextlib import asynccontextmanager, contextmanager
from importlib import import_module
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks, Header
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
from src.attributors.captum_smoothgrad import CaptumSmoothGradAttributor
from src.attributors.captum_gradcam import CaptumGradCamAttributor

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
MODEL_IDLE_TIMEOUT_ENV_VAR = "LUMIXAI_MODEL_IDLE_TIMEOUT_SEC"
DEFAULT_MODEL_IDLE_TIMEOUT_SEC = 1800.0
MODEL_IDLE_CHECK_INTERVAL_ENV_VAR = "LUMIXAI_MODEL_IDLE_CHECK_INTERVAL_SEC"
DEFAULT_MODEL_IDLE_CHECK_INTERVAL_SEC = 30.0
UNRECOVERABLE_CUDA_ERROR_MARKERS = (
    "device-side assert triggered",
)
UNRECOVERABLE_CUDA_RESTART_MESSAGE = (
    "Unrecoverable CUDA error detected. "
    "The backend is restarting automatically, so please retry in a few seconds."
)

_backend_restart_lock = threading.Lock()
_backend_restart_scheduled = False

def read_seconds_env(env_var: str, default_value: float, allow_zero: bool = False) -> float:
    raw_value = os.getenv(env_var, "").strip()
    if not raw_value:
        return default_value

    try:
        parsed_value = float(raw_value)
    except ValueError:
        print(f"Ignoring invalid {env_var}='{raw_value}'; falling back to {default_value}s.")
        return default_value

    if parsed_value < 0 or (parsed_value == 0 and not allow_zero):
        print(f"Ignoring out-of-range {env_var}='{raw_value}'; falling back to {default_value}s.")
        return default_value

    return parsed_value

def get_model_idle_timeout_sec() -> float:
    """Seconds a loaded model may sit unused before the reaper releases it. 0 disables it."""
    return read_seconds_env(MODEL_IDLE_TIMEOUT_ENV_VAR, DEFAULT_MODEL_IDLE_TIMEOUT_SEC, allow_zero=True)

def get_model_idle_check_interval_sec() -> float:
    """How often the reaper wakes up to compare the idle clock against the timeout."""
    return read_seconds_env(MODEL_IDLE_CHECK_INTERVAL_ENV_VAR, DEFAULT_MODEL_IDLE_CHECK_INTERVAL_SEC)

def format_duration(seconds: float) -> str:
    if seconds >= 120:
        return f"{round(seconds / 60)} minutes"
    return f"{round(seconds)} seconds"

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
    "captum_smoothgrad": {
        "name": "SmoothGrad (Captum)",
        "class": CaptumSmoothGradAttributor,
        "compatible_wrappers": IMAGE_CLASSIFICATION_WRAPPERS,
    },
    "captum_gradcam": {
        "name": "Grad-CAM (Captum)",
        "class": CaptumGradCamAttributor,
        "compatible_wrappers": IMAGE_CLASSIFICATION_WRAPPERS,
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
    "active_model_name": None,
    # Identifies the live configuration (model + attributor). Re-minted on every load and
    # attributor change, so a client holding an older token can be told its results would
    # not be about the model it thinks. See "SESSION OWNERSHIP".
    "config_id": None,
    # The session that loaded the configuration, and therefore holds the lease on it.
    # None means nobody claimed it (a client that sends no session header).
    "owner_session": None,
    # Describes the most recent teardown ({"reason", "model_name", "at"}), so /api/status
    # and /api/explain can tell the user *why* nothing is loaded. Cleared on a fresh load.
    "last_unload": None,
}

# --- 2.5 IDLE MODEL LIFECYCLE ---
# A loaded model holds its weights in (V)RAM for as long as it stays in `app_state`.
# `_last_activity_at` is the idle clock: every user-driven interaction with the model
# (load, attributor setup, explanation job) pushes it forward, and the reaper thread
# releases the model once the clock has been still for longer than the timeout.
# Read-only inspection (/api/status, job polling) deliberately does NOT count as usage,
# otherwise an open browser tab would keep the GPU pinned forever.
_activity_lock = threading.Lock()
_last_activity_at = time.monotonic()
# Loads and jobs in flight. While this is above zero the model is considered in use
# regardless of the idle clock.
_in_flight_operations = 0
# Attribution jobs alone. Tracked separately because a job is the one operation that must
# block a configuration change: swapping the model under a running job frees its weights
# mid-computation. A load in flight, by contrast, is itself a configuration change.
_in_flight_jobs = 0
_idle_reaper_stop = threading.Event()

def mark_activity() -> None:
    global _last_activity_at

    with _activity_lock:
        _last_activity_at = time.monotonic()

def begin_activity(is_job: bool = False) -> None:
    """Marks the start of an operation that must keep the model resident."""
    global _in_flight_operations, _in_flight_jobs, _last_activity_at

    with _activity_lock:
        _in_flight_operations += 1
        if is_job:
            _in_flight_jobs += 1
        _last_activity_at = time.monotonic()

def end_activity(is_job: bool = False) -> None:
    """Marks the end of such an operation and restarts the idle clock from now."""
    global _in_flight_operations, _in_flight_jobs, _last_activity_at

    with _activity_lock:
        _in_flight_operations = max(0, _in_flight_operations - 1)
        if is_job:
            _in_flight_jobs = max(0, _in_flight_jobs - 1)
        _last_activity_at = time.monotonic()

@contextmanager
def activity_scope(is_job: bool = False):
    begin_activity(is_job)
    try:
        yield
    finally:
        end_activity(is_job)

def get_idle_seconds() -> float:
    with _activity_lock:
        return time.monotonic() - _last_activity_at

def is_backend_busy() -> bool:
    with _activity_lock:
        return _in_flight_operations > 0

def are_jobs_in_flight() -> bool:
    with _activity_lock:
        return _in_flight_jobs > 0

def release_active_model(reason: str) -> bool:
    """Tears down the active wrapper/attributor and frees the memory they hold.

    Callers doing this while jobs may be running must hold `gpu_lock`, so that the model
    is never freed out from under an in-flight attribution.

    Args:
        reason (str): Why the model is going away ("manual", "idle_timeout", "model_switch").
            Surfaced through /api/status and in the /api/explain error message.

    Returns:
        bool: True if a model was actually released, False if nothing was loaded.
    """
    wrapper = app_state.get("active_wrapper")
    if wrapper is None:
        return False

    model_name = app_state.get("active_model_name")
    previous_device = getattr(wrapper, "device", None)
    # Drop the local handle too: the wrapper is only collectable once no reference remains.
    del wrapper

    print(f"Cleaning up memory and VRAM (reason: {reason})...")
    app_state["active_wrapper"] = None
    app_state["active_wrapper_name"] = None
    app_state["active_attributor"] = None
    app_state["active_attributor_id"] = None
    app_state["active_source"] = None
    app_state["active_model_name"] = None
    app_state["config_id"] = None
    app_state["owner_session"] = None
    app_state["last_unload"] = {
        "reason": reason,
        "model_name": model_name,
        "at": time.time(),
    }

    gc.collect()
    clear_cuda_memory(previous_device)
    return True

def maybe_release_idle_model() -> None:
    """Releases the active model if it has gone unused for longer than the idle timeout."""
    idle_timeout_sec = get_model_idle_timeout_sec()
    if idle_timeout_sec <= 0:
        return

    expected_wrapper = app_state.get("active_wrapper")
    if expected_wrapper is None or is_backend_busy() or get_idle_seconds() < idle_timeout_sec:
        return

    if not gpu_lock.acquire(blocking=False):
        # A job is on the GPU right now; it will refresh the idle clock when it finishes.
        return

    try:
        # Re-check under the lock: the user may have submitted work while we were deciding.
        idle_seconds = get_idle_seconds()
        if is_backend_busy() or idle_seconds < idle_timeout_sec:
            return

        # /api/load swaps this slot without taking `gpu_lock`, so confirm we are about to
        # release the same model we judged idle and not one that just replaced it.
        if app_state.get("active_wrapper") is not expected_wrapper:
            return

        model_name = app_state.get("active_model_name")
        del expected_wrapper
        if release_active_model(reason="idle_timeout"):
            print(f"Released '{model_name}' after {format_duration(idle_seconds)} of inactivity.")
    finally:
        gpu_lock.release()

def model_idle_reaper_loop() -> None:
    while not _idle_reaper_stop.wait(get_model_idle_check_interval_sec()):
        try:
            maybe_release_idle_model()
        except Exception:
            # Never let a transient failure kill the reaper: the model would then stay
            # resident for the rest of the process lifetime with no way back.
            traceback.print_exc()

@asynccontextmanager
async def lifespan(app: FastAPI):
    idle_timeout_sec = get_model_idle_timeout_sec()
    if idle_timeout_sec > 0:
        print(f"Idle model reaper enabled: unloading after {format_duration(idle_timeout_sec)} of inactivity.")
    else:
        print(f"Idle model reaper disabled ({MODEL_IDLE_TIMEOUT_ENV_VAR}=0). Models stay resident until unloaded.")

    reaper_thread = threading.Thread(target=model_idle_reaper_loop, name="model-idle-reaper", daemon=True)
    reaper_thread.start()
    try:
        yield
    finally:
        _idle_reaper_stop.set()
        reaper_thread.join(timeout=5)

# --- 2.6 SESSION OWNERSHIP ---
# The backend holds one model, but the frontend, notebooks and experiment scripts can all
# reach it at once. Without a notion of "whose model is this", a second client loading a
# model would silently redirect everyone else's attributions onto it — producing results
# that look plausible but explain the wrong model, recorded in history under the old name.
#
# Two rules keep that from happening, and they answer different questions:
#   * the config token answers "are these results about the model the client asked for?"
#     Clients pass `config_id` on /api/explain; a mismatch is refused, never guessed at.
#   * the lease answers "may this client change what is loaded?" The session that loaded
#     the model keeps that right until the model falls idle, so nobody's configuration is
#     replaced out from under them by accident. Taking over is possible, but deliberate.
# Reading and running attributions is never gated: clients share the loaded model freely,
# which is the point of a shared demo box. Only mutations are owned.
SESSION_HEADER = "X-LumiXAI-Session"
# How long to wait for the GPU before concluding that a job has claimed it and this
# configuration change should back off rather than block the HTTP request.
GPU_LOCK_ACQUIRE_TIMEOUT_SEC = 5.0

def get_session_lease_ttl_sec() -> float:
    """How long an unused configuration stays claimed by the session that loaded it.

    Tied to the idle timeout: a lease has no reason to outlive the model it protects.
    When the reaper is disabled the model never expires on its own, so the lease falls
    back to the default idle window rather than becoming a permanent claim.
    """
    idle_timeout_sec = get_model_idle_timeout_sec()
    return idle_timeout_sec if idle_timeout_sec > 0 else DEFAULT_MODEL_IDLE_TIMEOUT_SEC

def get_lease_seconds_remaining() -> Optional[float]:
    """Seconds before the active configuration is up for grabs, or None if it is unclaimed."""
    if app_state.get("active_wrapper") is None or app_state.get("owner_session") is None:
        return None

    if is_backend_busy():
        return get_session_lease_ttl_sec()

    return max(0.0, get_session_lease_ttl_sec() - get_idle_seconds())

def is_lease_held_by_other(session_id: Optional[str]) -> bool:
    """Whether another session's claim currently blocks `session_id` from changing the config."""
    owner_session = app_state.get("owner_session")
    if owner_session is None or app_state.get("active_wrapper") is None:
        return False
    if session_id is not None and session_id == owner_session:
        return False

    seconds_remaining = get_lease_seconds_remaining()
    return seconds_remaining is not None and seconds_remaining > 0

def build_lease_conflict_error() -> str:
    model_name = app_state.get("active_model_name") or "A model"

    if are_jobs_in_flight():
        return (
            f"'{model_name}' was loaded by another session and is running a job right now. "
            "Wait for it to finish, or take over to load your own configuration."
        )

    idle_text = format_duration(get_idle_seconds())
    remaining_text = format_duration(get_lease_seconds_remaining() or 0)
    return (
        f"'{model_name}' is loaded by another session and was last used {idle_text} ago. "
        f"It is released automatically in {remaining_text}. Take over to load your configuration now."
    )

def build_stale_config_error() -> str:
    model_name = app_state.get("active_model_name")
    now_holding = f"'{model_name}'" if model_name else "a different configuration"
    return (
        f"Your configuration is no longer the one loaded on the backend, which now holds {now_holding}. "
        "Load your configuration again to continue."
    )

def claim_configuration(session_id: Optional[str]) -> str:
    """Mints a token for the configuration just set up and records who owns it."""
    config_id = uuid.uuid4().hex
    app_state["config_id"] = config_id
    app_state["owner_session"] = session_id
    return config_id

def require_config_control(session_id: Optional[str], force: bool) -> None:
    """Guards every configuration change. Raises if the caller may not proceed.

    Args:
        session_id (Optional[str]): Caller identity from the `X-LumiXAI-Session` header.
        force (bool): Whether the caller explicitly chose to take over another session's
            configuration. Never bypasses a running job: a takeover reassigns the claim,
            it does not pull weights out from under work in progress.
    """
    if not force and is_lease_held_by_other(session_id):
        raise HTTPException(423, build_lease_conflict_error())

    if are_jobs_in_flight():
        raise HTTPException(
            409,
            "A job is running on the backend right now. Wait for it to finish, then try again.",
        )

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
    # Take over a configuration another session still holds. The frontend sets this only
    # after the user confirms the takeover it was warned about.
    force: bool = False

class AttributorRequest(BaseModel):
    attributor_id: str
    params: Optional[Dict[str, Any]] = {}
    force: bool = False

class UnloadRequest(BaseModel):
    force: bool = False

class ExplainRequest(BaseModel):
    # The configuration these results are meant to be about, as returned by /api/load and
    # /api/set_attributor. Omitting it means "whatever is loaded", which is what the pre-token
    # clients expect; passing it turns a model swap into a refusal instead of wrong results.
    config_id: Optional[str] = None
    text: Optional[str] = None
    # Base64-encoded input image, used instead of `text` for image classification models.
    image_base64: Optional[str] = None
    # Original file name of the uploaded image, used to label the job in history
    # instead of a generic placeholder.
    image_filename: Optional[str] = None
    target_class: Optional[int] = None
    ignore_special_tokens: bool = True
    seed: Optional[int] = None
    max_new_tokens: Optional[int] = None
    disable_thinking: bool = False
    # Whether to wrap text-generation prompts in the model's chat template
    use_chat_template: bool = True
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
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 5. BACKGROUND ---
def run_explanation_task(job_id: str, text: Optional[str], target_class: Optional[int], ignore_special_tokens: bool = True, seed: Optional[int] = None, guidance_scale: Optional[float] = None, negative_prompt: Optional[str] = None, disable_thinking: bool = False, max_new_tokens: Optional[int] = None, image_base64: Optional[str] = None, use_chat_template: bool = True, expected_config_id: Optional[str] = None):
    """Executes the XAI attribution logic asynchronously.

    This function runs in a separate thread. It acquires the global `gpu_lock` to ensure
    that only one heavy inference process hits the GPU at a time, preventing Out-Of-Memory
    errors or DAAM tracing conflicts. Upon completion, it formats the payload and updates
    the SQLite database.

    The whole job also runs inside an `activity_scope`, which keeps the idle reaper from
    unloading the model while the job is queued or running, and restarts the idle clock
    when it finishes rather than when it was submitted.

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
        expected_config_id (Optional[str], optional): The configuration this job was
            submitted against. Re-checked here because the job may have waited in the
            queue while the configuration changed underneath it.
    """
    with activity_scope(is_job=True), gpu_lock:
        # Start the clock only after acquiring the GPU lock, so execution_time_sec
        # reflects the actual compute time and not the time spent queued behind
        # other jobs waiting for the lock.
        start_time = time.time()
        try:
            # The submitted-against configuration must still be the live one. Between
            # submission and this point another session may have loaded its own model, and
            # attributing to that one would answer a question nobody asked.
            if expected_config_id is not None and app_state.get("config_id") != expected_config_id:
                raise ValueError(build_stale_config_error())

            attributor = app_state["active_attributor"]
            wrapper = app_state["active_wrapper"]

            if not attributor or not wrapper:
                raise ValueError("Modello o Attributor disconnessi durante l'esecuzione")

            # Applied under gpu_lock, so mutating the shared wrapper here is safe: jobs run
            # one at a time. Only text-generation wrappers expose this attribute.
            if hasattr(wrapper, "use_chat_template"):
                wrapper.use_chat_template = use_chat_template

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
def load_model(req: LoadRequest, session_id: Optional[str] = Header(None, alias=SESSION_HEADER)):
    """Loads a model into the global application state.

    This endpoint automatically identifies the task type (e.g., text-generation vs image-generation)
    via the Hugging Face API and instantiates the correct Wrapper class. It also aggressively
    cleans the VRAM before loading to prevent memory overflows.

    Loading replaces the single model the backend holds, so it is refused (423) while another
    session's lease is alive unless `force` is set, and (409) while any job is running.
    """
    require_config_control(session_id, req.force)

    with activity_scope():
        # Hold the GPU across teardown and load: jobs take the same lock, so this is what
        # stops a queued job from starting against a half-swapped model — the free() and the
        # forward pass would otherwise interleave.
        if not gpu_lock.acquire(timeout=GPU_LOCK_ACQUIRE_TIMEOUT_SEC):
            raise HTTPException(409, "A job just claimed the GPU. Wait for it to finish, then load again.")

        try:
            return load_model_into_state(req, session_id)
        finally:
            gpu_lock.release()

def load_model_into_state(req: LoadRequest, session_id: Optional[str] = None):
    """Runs the actual load.

    Split out of the endpoint so that `activity_scope` covers the whole operation: a
    download can take minutes, and the reaper must not read that stretch of silence as
    idleness and free the model just as it lands.
    """
    try:
        release_active_model(reason="model_switch")

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
        app_state["last_unload"] = None
        config_id = claim_configuration(session_id)

        return {
            "status": "loaded", "model": req.model_name,
            "wrapper": wrapper_name, "device": real_device,
            "detected_task": detected_task,
            "config_id": config_id
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
def unload_model(
    req: Optional[UnloadRequest] = None,
    session_id: Optional[str] = Header(None, alias=SESSION_HEADER),
):
    """Releases VRAM and clears the global state. Protected by the GPU Mutex lock.

    Unloading is a configuration change like any other, so another session's live lease
    protects the model from it unless `force` is set.
    """
    require_config_control(session_id, bool(req and req.force))

    with gpu_lock:
        try:
            if release_active_model(reason="manual"):
                return {"status": "success", "message": "Cleaned up memory and VRAM. Model unloaded."}
            return {"status": "success", "message": "No model in memory to unload."}
        except Exception as e:
            raise HTTPException(500, f"Error occurred while cleaning up memory: {str(e)}")

@app.get("/api/status")
def get_status(session_id: Optional[str] = Header(None, alias=SESSION_HEADER)):
    """Reports what is actually resident in memory, who holds it, and how long it has left.

    The frontend polls this so its view of the configuration cannot drift from the
    backend's — after the idle reaper releases a model, or after another session loads
    one. Polling is deliberately free of side effects: it does not touch the idle clock,
    which would otherwise let an idle browser tab pin the GPU forever.
    """
    active_wrapper = app_state.get("active_wrapper")
    idle_timeout_sec = get_model_idle_timeout_sec()
    idle_seconds = get_idle_seconds()

    seconds_until_unload = None
    if active_wrapper is not None and idle_timeout_sec > 0:
        seconds_until_unload = max(0.0, idle_timeout_sec - idle_seconds)

    owned_by_you = bool(
        active_wrapper is not None
        and session_id is not None
        and app_state.get("owner_session") == session_id
    )
    lease_seconds_remaining = get_lease_seconds_remaining()

    return {
        "model_loaded": active_wrapper is not None,
        "model_name": app_state.get("active_model_name"),
        "source": app_state.get("active_source"),
        "wrapper": app_state.get("active_wrapper_name"),
        "device": getattr(active_wrapper, "device", None),
        "attributor_id": app_state.get("active_attributor_id"),
        "busy": is_backend_busy(),
        "idle_seconds": round(idle_seconds, 1),
        "idle_timeout_sec": idle_timeout_sec if idle_timeout_sec > 0 else None,
        "seconds_until_unload": round(seconds_until_unload, 1) if seconds_until_unload is not None else None,
        "last_unload": app_state.get("last_unload"),
        # Ownership is reported as an answer about *you*, never as another session's id.
        "config_id": app_state.get("config_id"),
        "owned_by_you": owned_by_you,
        "held_by_other_session": is_lease_held_by_other(session_id),
        "lease_seconds_remaining": round(lease_seconds_remaining, 1) if lease_seconds_remaining is not None else None,
    }

@app.post("/api/set_attributor")
def set_attributor(req: AttributorRequest, session_id: Optional[str] = Header(None, alias=SESSION_HEADER)):
    """Instantiates and attaches an Attributor algorithm to the currently loaded model.

    The attributor is part of the configuration, so this mints a fresh `config_id`: clients
    holding the previous token are no longer explaining the same thing.
    """
    if not app_state["active_wrapper"]:
        raise HTTPException(400, build_no_active_model_error())
    if req.attributor_id not in AVAILABLE_ATTRIBUTORS:
        raise HTTPException(400, "Attributor ID not found")

    require_config_control(session_id, req.force)

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
        with activity_scope():
            AttrClass = resolve_attributor_class(req.attributor_id)
            app_state["active_attributor"] = AttrClass(app_state["active_wrapper"])
            app_state["active_attributor_id"] = req.attributor_id
            config_id = claim_configuration(session_id)
        return {
            "status": "active",
            "id": req.attributor_id,
            "name": AVAILABLE_ATTRIBUTORS[req.attributor_id]["name"],
            "config_id": config_id,
        }
    except Exception as e:
        raise HTTPException(400, f"Attributor '{req.attributor_id}' is not currently available: {e}")


def build_no_active_model_error() -> str:
    """Explains why no model is loaded, distinguishing an idle unload from a fresh start."""
    last_unload = app_state.get("last_unload") or {}

    if last_unload.get("reason") == "idle_timeout":
        model_name = last_unload.get("model_name") or "The model"
        idle_timeout_sec = get_model_idle_timeout_sec()
        idle_timeout_text = format_duration(idle_timeout_sec) if idle_timeout_sec > 0 else "a period"
        return (
            f"'{model_name}' was unloaded to free memory after {idle_timeout_text} of inactivity. "
            "Load the configuration again to continue."
        )

    return "Model and attributor must be loaded first."


@app.post("/api/explain", response_model=JobResponse)
def explain(req: ExplainRequest, background_tasks: BackgroundTasks):
    """Enqueues a new asynchronous XAI job.

    Running attributions is not gated on the lease: clients share the loaded model. What is
    checked is `config_id`, so results are never quietly produced by a model the caller
    didn't ask for.
    """
    if not app_state.get("active_attributor") or not app_state.get("active_wrapper"):
        raise HTTPException(400, build_no_active_model_error())

    # A client that named a configuration gets told when it is gone, rather than quietly
    # having its question answered by whatever model happens to be loaded now.
    active_config_id = app_state.get("config_id")
    if req.config_id is not None and req.config_id != active_config_id:
        raise HTTPException(409, build_stale_config_error())

    # Submitting work counts as usage right away, so the reaper cannot release the model
    # in the gap between this response and the background task picking the job up.
    mark_activity()

    source_name = app_state.get("active_source", "unknown")
    model_name = app_state.get("active_model_name", "unknown")
    active_attributor_id = app_state.get("active_attributor_id")
    attributor_name = AVAILABLE_ATTRIBUTORS.get(active_attributor_id, {}).get("name", "Unknown")

    if req.max_new_tokens is not None and req.max_new_tokens <= 0:
        raise HTTPException(400, "max_new_tokens must be a positive integer.")

    if not req.text and not req.image_base64:
        raise HTTPException(400, "Either 'text' or 'image_base64' must be provided.")

    job_prompt = req.text if req.text else (req.image_filename or "[Uploaded Image]")
    job_id = create_job(job_prompt, source_name, model_name, attributor_name)

    # Pin the job to the configuration live at submission even when the client sent no
    # token: the job record above is stamped with this model's name, and it must not end up
    # describing an attribution computed on whatever replaced it while the job queued.
    expected_config_id = req.config_id or active_config_id

    background_tasks.add_task(run_explanation_task, job_id, req.text, req.target_class, req.ignore_special_tokens, req.seed, req.guidance_scale, req.negative_prompt, req.disable_thinking, req.max_new_tokens, req.image_base64, req.use_chat_template, expected_config_id)

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
