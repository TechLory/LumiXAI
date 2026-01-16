from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from src.utils.hf_hub import search_hf_models
from huggingface_hub import HfApi

# --- IMPORTS ---
from src.wrappers.hf_text import HFTextWrapper
from src.attributors.captum_grad import CaptumGradientsAttributor

# --- 1. REGISTRY ---
AVAILABLE_WRAPPERS = {
    "hf_text": HFTextWrapper,
}

AVAILABLE_SOURCES = [
    {
        "id": "huggingface", 
        "name": "Hugging Face Hub", 
        "type": "remote"
    },
    {
        "id": "custom_wrapper", 
        "name": "Custom Model (Local) -DEMO-", 
        "type": "local"
    }
]

AVAILABLE_ATTRIBUTORS = {
    "captum_ig": {
        "name": "Integrated Gradients (Captum)", 
        "class": CaptumGradientsAttributor
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
    target: int
    tokens: List[str]
    scores: List[float]

# --- 4. SETUP APP ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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
                detected_task = "text-classification" # Failed: assuming text-classification

            # 2. Logical Switch (Dispatcher)
            match detected_task:
                case "text-classification" | "fill-mask" | "token-classification" | "question-answering" | "summarization" | "translation" | "text-generation":
                    from src.wrappers.hf_text import HFTextWrapper
                    wrapper_instance = HFTextWrapper(req.model_name, req.device)
                    wrapper_name = "hf_text"
            
                case "text-to-image" | "image-classification":
                    # from src.wrappers.hf_image import HFImageWrapper
                    # wrapper_instance = HFImageWrapper(req.model_name, req.device)
                    # wrapper_name = "hf_image"
                    pass

                case _:
                    # Fallback to generic text wrapper
                    from src.wrappers.hf_text import HFTextWrapper
                    wrapper_instance = HFTextWrapper(req.model_name, req.device)
                    wrapper_name = "hf_text (fallback)"

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
    """Generate explanation for the input text"""
    attr = app_state["active_attributor"]
    if not attr:
        raise HTTPException(400, "No attributor active")
    
    # Call the .attribute() method
    output = attr.attribute(req.text, req.target_class)
    
    # Convert the output to JSON-friendly list
    return {
        "target": output.target,
        # Assume input_features are ordered
        "tokens": [f.content for f in output.input_features],
        "scores": output.heatmap.tolist() # Convert numpy array to list
    }