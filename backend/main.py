from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any, Dict

# --- IMPORTS ---
from src.wrappers.hf_text import HFTextWrapper
from src.attributors.captum_grad import CaptumGradientsAttributor
from src.utils.hf_hub import search_hf_models

# --- 1. REGISTRY ---
AVAILABLE_WRAPPERS = {
    "hf_text": {"name": "Hugging Face (Transformers)", "class": HFTextWrapper},
    "hf_diffusers": {"name": "Hugging Face (Diffusers)", "class": HFTextWrapper},
    "test1": {"name": "Test Wrapper 1", "class": HFTextWrapper},
    "test2": {"name": "Test Wrapper 2", "class": HFTextWrapper},
}

AVAILABLE_ATTRIBUTORS = {
    "captum_integrated_gradients": CaptumGradientsAttributor,
    # "daam_attention": DaamAttributor (next)
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
    wrapper_id: str      # Es: "hf_text" rivedere id!!!1
    model_name: str      # Es: "distilbert..."
    device: str = "cpu"

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
    Return the available wrappers and attributors.
    """
    return {
        "wrappers": [{"id": key, "name": value["name"]} for key, value in AVAILABLE_WRAPPERS.items()],
        "attributors": [{"id": key, "name": value.__name__} for key, value in AVAILABLE_ATTRIBUTORS.items()]
    }

@app.get("/api/search", response_model=List[SearchResult])
def search_models(source: str, q: str):
    """
    Endpoint for search (Google-style)
    """

    print(f"Search request - source: {source}, query: {q}")

    # QUI SWITCH SULLE VARIE FONTI DI MODELLI
    # TODO
    results = search_hf_models(query=q, limit=20)
    return results

@app.post("/api/load")
def load_model(req: LoadRequest):
    """Load the model in memory"""
    try:
        # 1. Retrieve the class from the Registry
        if req.wrapper_id not in AVAILABLE_WRAPPERS:
            raise HTTPException(400, "Wrapper ID not found")
        
        WrapperClass = AVAILABLE_WRAPPERS[req.wrapper_id]
        
        # 2. Instantiate the class
        print(f"Loading {req.model_name} with {req.wrapper_id}...")
        wrapper_instance = WrapperClass(req.model_name, req.device)
        
        # 3. Save in state
        app_state["active_wrapper"] = wrapper_instance
        
        # Reset attributor when the model changes
        app_state["active_attributor"] = None 
        
        return {"status": "loaded", "model": req.model_name}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(500, str(e))

@app.post("/api/set_attributor")
def set_attributor(attributor_id: str):
    """Set active attributor"""
    if not app_state["active_wrapper"]:
        raise HTTPException(400, "Load a model first!")
    
    if attributor_id not in AVAILABLE_ATTRIBUTORS:
        raise HTTPException(400, "Attributor ID not found")

    AttrClass = AVAILABLE_ATTRIBUTORS[attributor_id]
    
    # Instantiate the attributor passing the active wrapper
    app_state["active_attributor"] = AttrClass(app_state["active_wrapper"])
    
    return {"status": "attributor_set", "id": attributor_id}

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