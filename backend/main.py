import os
import torch
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
HF_CACHE_DIR = BASE_DIR / "hf_models_cache"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
print(f"hf cache dir set to: {HF_CACHE_DIR}")
os.environ["HF_HOME"] = str(HF_CACHE_DIR)


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from src.utils.hf_hub import search_hf_models
from huggingface_hub import HfApi

# --- IMPORTS ---
from src.wrappers.hf_text_classification import HFTextClassificationWrapper
from src.wrappers.hf_text_generation import HFTextGenerationWrapper
from src.wrappers.hf_image import HFImageWrapper

from src.attributors.daam import DAAMAttributor
from src.attributors.captum_grad import CaptumGradientsAttributor

def get_optimal_device(requested_device: str = "auto") -> str:
    if requested_device == "cpu":
        return "cpu"

    # 1. Check CUDA (Nvidia)
    if torch.cuda.is_available():
        if requested_device in ["auto", "cuda"]:
            return "cuda"
    
    # 2. Check MPS (Mac Apple Silicon)
    if torch.backends.mps.is_available():
        if requested_device in ["auto", "cuda", "mps"]:
            return "mps"

    # 3. Fallback CPU
    print(f"Requested '{requested_device}' but GPU not found. Falling back to CPU.")
    return "cpu"

# --- 1. REGISTRY ---
AVAILABLE_WRAPPERS = {
    "hf_text_classification": HFTextClassificationWrapper,
    "hf_text_generation": HFTextGenerationWrapper,
    "hf_image": HFImageWrapper,
}

AVAILABLE_SOURCES = [
    {
        "id": "huggingface", 
        "name": "Hugging Face Hub", 
        "type": "remote"
    },
]

AVAILABLE_ATTRIBUTORS = {
    "captum_ig": {
        "name": "Integrated Gradients (Captum)", 
        "class": CaptumGradientsAttributor
    },
    "daam": {
        "name": "DAAM (Diffusion Attentive Attribution Maps)", 
        "class": DAAMAttributor
    },
}

# --- 2. GLOBAL STATE ---
app_state: Dict[str, Any] = {
    "active_wrapper": None,
    "active_attributor": None
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

class ExplanationResponse(BaseModel):
    target_id: Any            # token ID /  "image_generation"
    predicted_token: Optional[str] = None
    tokens: List[str]
    scores: Any               # List[float] / Base64
    generated_image: Optional[str] = None #

# --- 4. SETUP APP ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    # allow_origins=["http://localhost:3000"],
    allow_origins=["*"], # Allow all origins for development; restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 5. ENDPOINTS ---

@app.get("/")
def health_check():
    """Endpoint to check if the backend is online"""
    return {"status": "XAI Backend Online", "version": "0.1.0"}

@app.get("/api/manifest")
def get_manifest():
    """
    Return the available sources and attributors.
    """
    return {
        "sources": AVAILABLE_SOURCES,
        "attributors": [
            {"id": key, "name": value["name"]} 
            for key, value in AVAILABLE_ATTRIBUTORS.items()
        ]
    }

@app.get("/api/search", response_model=List[SearchResult])
def search_models(source: str, q: str):
    """
    Endpoint for search.
    Dispatches to the appropriate search function based on source.
    """

    print(f"Search request - source: {source}, query: {q}")

    match source:
        case "huggingface":
            return search_hf_models(query=q, limit=10)
        case "custom_demo":
            ## CUSTOM IMPLEMENTATION
            return []
        case _:
            print(f"Unknown source: {source}")
            return []

@app.post("/api/load")
def load_model(req: LoadRequest):
    """
    Dispatcher
    """
    try:
        real_device = get_optimal_device(req.device)

        wrapper_instance = None
        wrapper_name = "unknown"
        detected_task = "unknown"

        ### HUGGING FACE HUB
        if req.source == "huggingface":            
            # 1. get model type info from HF Hub
            api = HfApi()
            try:
                info = api.model_info(req.model_name)
                detected_task = info.pipeline_tag
            except Exception:
                detected_task = "-" # Failed: assuming text-classification (fallback)

            # 2. Logical Switch (Dispatcher)
            match detected_task:
                case "text-classification" | "fill-mask" | "token-classification":
                    wrapper_instance = HFTextClassificationWrapper(req.model_name, real_device)
                    wrapper_name = "hf_text_classification"
            
                case "text-generation" | "text2text-generation" | "translation" | "summarization":
                    wrapper_instance = HFTextGenerationWrapper(req.model_name, real_device) # pyright: ignore[reportAbstractUsage]
                    wrapper_name = "hf_text_generation"

                case "text-to-image":
                    wrapper_instance = HFImageWrapper(req.model_name, real_device)
                    wrapper_name = "hf_image"

                case _:
                    # Fallback to text-classification
                    wrapper_instance = HFTextClassificationWrapper(req.model_name, real_device)
                    wrapper_name = "hf_text_classification (fallback)"

        ### OTHER SOURCES...
        #### CUSTOM WRAPPER (DEMO)
        elif req.source == "custom_wrapper":
            # from src.wrappers.custom_wrapper import CustomWrapper
            # wrapper_instance = CustomWrapper(req.model_name, req.device)
            raise HTTPException(501, "Custom Wrapper not available at the moment")

        ### UNSUPPORTED SOURCE
        else:
            raise HTTPException(400, f"Source '{req.source}' not supported")

        # SET GLOBAL STATE
        app_state["active_wrapper"] = wrapper_instance
        app_state["active_attributor"] = None
        
        return {
            "status": "loaded",
            "model": req.model_name,
            "wrapper": wrapper_name,
            "device": real_device,
            "detected_task": detected_task if req.source == "huggingface" else "custom"
        }

    except Exception as e:
        print(f"Load Error: {e}")
        raise HTTPException(500, str(e))

@app.post("/api/set_attributor")
def set_attributor(req: AttributorRequest):
    """
    Set the active attributor.
    Requires an active model wrapper.
    """

    print(req.attributor_id)

    if not app_state["active_wrapper"]:
        raise HTTPException(400, "No model loaded.")
    
    if req.attributor_id not in AVAILABLE_ATTRIBUTORS:
        raise HTTPException(400, f"Attributor ID ({req.attributor_id}) not found")

    try:
        attributor_entry = AVAILABLE_ATTRIBUTORS[req.attributor_id]
        AttrClass = attributor_entry["class"]
            
        active_wrapper = app_state["active_wrapper"]
        attributor_instance = AttrClass(active_wrapper)

        app_state["active_attributor"] = attributor_instance

        return {
            "status": "active", 
            "id": req.attributor_id, 
            "name": attributor_entry["name"]
        }
    
    except Exception as e:
        print(f"(backend) Error setting attributor: {e}")
        raise HTTPException(500, str(e))

@app.post("/api/explain", response_model=ExplanationResponse)
def explain(req: ExplainRequest):
    """Generate explanation for the input text or image"""
    
    attributor = app_state["active_attributor"]
    if not attributor:
        raise HTTPException(400, "No attributor active")
    
    wrapper = app_state["active_wrapper"]
    try:
        output = attributor.attribute(req.text, req.target_class)
        predicted_word = None
        
        # TEXT GENERATION (Target: string)
        if output.target == "text_generation":
            return {
                "target_id": "text_generation",
                "predicted_token": None,
                "tokens": [], 
                "scores": output.heatmap,
                "generated_image": None
            }

        # CLASSIFICATION/IMAGE (Target: int or None)
        if hasattr(wrapper, "tokenizer") and isinstance(output.target, int):
            try:
                predicted_word = wrapper.tokenizer.decode([output.target])
            except:
                pass

        return {
            "target_id": output.target,
            "predicted_token": predicted_word,
            "tokens": [f.content for f in output.input_features],
            "scores": output.heatmap.tolist() if hasattr(output.heatmap, "tolist") else output.heatmap,
            "generated_image": output.generated_image
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))