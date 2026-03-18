"""Hugging Face Hub integration utilities.

This module provides helper functions to interact with the external Hugging Face Hub API,
facilitating model search and metadata retrieval for the framework's frontend.
"""

from typing import List, Dict, Any
from huggingface_hub import HfApi

def search_hf_models(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Searches the Hugging Face Hub for models matching the user's query.
    
    This function filters models based on the query string and orders the results 
    by the number of downloads to ensure the most relevant and popular models appear first.
    It performs a lightweight search (`full=False`) to minimize API latency.

    Args:
        query (str): The search string (e.g., "bert", "stable-diffusion"). 
            Must be at least 2 characters long.
        limit (int, optional): The maximum number of results to return. Defaults to 10.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing model metadata. Each dictionary includes:
            - `id` (str): The official Hugging Face model ID.
            - `task` (str): The pipeline tag (e.g., 'text-classification', 'text-to-image').
            - `likes` (int): The number of likes the model has received.
            - `downloads` (int): The number of times the model has been downloaded.
            Returns an empty list if the query is too short or if the API call fails.
    """
    api = HfApi()
    
    if not query or len(query.strip()) < 2:
        return []

    try:
        models = api.list_models(
            search=query,
            sort="downloads",
            direction=-1,
            limit=limit,
            full=False 
        )

        results = []
        for m in models:
            results.append({
                "id": m.modelId, # type: ignore
                "task": m.pipeline_tag if m.pipeline_tag else "unknown",
                "likes": m.likes,
                "downloads": m.downloads
            })
            
        return results
    
    except Exception as e:
        print(f"Error searching HF Hub: {e}")
        return []