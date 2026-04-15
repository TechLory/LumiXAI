"""Hugging Face Hub integration utilities.

This module provides helper functions to interact with the external Hugging Face Hub API,
facilitating model search and metadata retrieval for the framework's frontend.
"""

from typing import List, Dict, Any, Optional
from huggingface_hub import HfApi
from .hf_auth import has_hf_token, hf_auth_kwargs

def get_model_access_issue(model_info: Any) -> Optional[str]:
    """Returns a human-friendly access error for restricted Hugging Face repos."""
    if getattr(model_info, "private", False):
        return "private"
    if getattr(model_info, "gated", False):
        return "gated"
    if getattr(model_info, "disabled", False):
        return "disabled"
    return None

def build_model_access_error(model_id: str, access_issue: str) -> str:
    if access_issue == "gated":
        if has_hf_token():
            return (
                f"Model '{model_id}' is gated on Hugging Face. "
                "Make sure the account behind HF_TOKEN has been granted access on the model page."
            )
        return (
            f"Model '{model_id}' is gated on Hugging Face. "
            "Request access on the model page and set HF_TOKEN for the backend container."
        )
    if access_issue == "private":
        if has_hf_token():
            return (
                f"Model '{model_id}' is private on Hugging Face. "
                "Make sure the account behind HF_TOKEN has access to this repository."
            )
        return (
            f"Model '{model_id}' is private on Hugging Face. "
            "Set HF_TOKEN for an account that has access to this repository."
        )
    if access_issue == "disabled":
        return (
            f"Model '{model_id}' is disabled on Hugging Face and cannot be loaded. "
            "Please choose another model."
        )
    return f"Model '{model_id}' is not accessible from this backend."

def build_hf_load_error(model_id: str, error: Exception) -> str:
    error_text = str(error)
    normalized_error = error_text.lower()

    if "gated repo" in normalized_error or "cannot access gated repo" in normalized_error:
        return build_model_access_error(model_id, "gated")
    if "is restricted" in normalized_error and "please log in" in normalized_error:
        return build_model_access_error(model_id, "gated")
    if "401 client error" in normalized_error and "resolve/main/config.json" in normalized_error:
        return build_model_access_error(model_id, "gated")
    if "403 client error" in normalized_error and "resolve/main/config.json" in normalized_error:
        return build_model_access_error(model_id, "gated")
    if "repository not found" in normalized_error or "404 client error" in normalized_error:
        return (
            f"Model '{model_id}' was not found on Hugging Face. "
            "Please verify the model ID."
        )

    return error_text


def is_model_access_blocked(access_issue: Optional[str]) -> bool:
    if access_issue is None:
        return False
    if access_issue == "disabled":
        return True
    if access_issue in {"gated", "private"}:
        return not has_hf_token()
    return True

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
    api = HfApi(**hf_auth_kwargs())
    
    if not query or len(query.strip()) < 2:
        return []

    try:
        models = api.list_models(
            search=query,
            sort="downloads",
            direction=-1,
            limit=max(limit * 4, 20),
            full=True
        )

        results = []
        for m in models:
            access_issue = get_model_access_issue(m)
            if is_model_access_blocked(access_issue):
                continue

            results.append({
                "id": m.modelId, # type: ignore
                "task": m.pipeline_tag if m.pipeline_tag else "unknown",
                "likes": m.likes,
                "downloads": m.downloads
            })

            if len(results) >= limit:
                break
            
        return results
    
    except Exception as e:
        print(f"Error searching HF Hub: {e}")
        return []
