from typing import List, Dict
from huggingface_hub import HfApi

def search_hf_models(query: str, limit: int = 10) -> List[Dict[str, str]]:
    """
    Search HF models and return useful metadata.
    Docs: https://huggingface.co/docs/huggingface_hub/package_reference/hf_api
    
    Returns:
        Dictionary : [{'id': '...', 'task': '...', 'likes': 1200}, ...]
    """
    api = HfApi()
    
    if not query or len(query.strip()) < 2:
        return []

    try:
        # Search models ordered by downloads
        models = api.list_models(
            search=query,
            sort="downloads",
            direction=-1,
            limit=limit,
            full=False # (Less data to speed up the response)
        )

        results = []
        for m in models:
            results.append({
                "id": m.modelId, # type: ignore
                # The task (e.g., 'text-classification'):
                "task": m.pipeline_tag if m.pipeline_tag else "unknown",
                "likes": m.likes,
                "downloads": m.downloads
            })
            
        return results
    
    except Exception as e:
        print(f"Error searching HF Hub: {e}")
        return []