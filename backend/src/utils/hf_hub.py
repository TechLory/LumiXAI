"""
Hugging Face Hub Utilities Module

This module provides utility functions for interacting with the
Hugging Face Hub API, including model search functionality.
"""

from typing import List, Optional
from huggingface_hub import HfApi


def search_hf_models(
    search_query: str, 
    task_filter: Optional[str] = None, 
    limit: int = 10
) -> List[str]:
    """
    Search for models on the Hugging Face Hub.

    Queries the Hugging Face Hub API to retrieve a list of model identifiers
    matching the specified search criteria, sorted by download count.

    Args:
        search_query: The search term to filter models by name or description.
        task_filter: Optional task type filter (e.g., "text-classification",
            "image-classification"). Accepts a string or list of tags.
        limit: Maximum number of results to return. Defaults to 10.

    Returns:
        A list of model identifiers (strings) matching the search criteria.
        Returns an empty list if an error occurs during the API call.
    """
    # Initialize the Hugging Face Hub API client.
    api = HfApi()
    
    try:
        # Query the Hub for models matching the specified criteria.
        models = api.list_models(
            search=search_query,
            # Apply optional task-based filtering.
            filter=task_filter,
            # Sort results by popularity (download count).
            sort="downloads",
            # Use descending order to prioritize most downloaded models.
            direction=-1,
            limit=limit
        )
        
        # Extract and return the model identifiers from the response.
        # Note: list_models returns a generator or list of ModelInfo objects.
        return [model.modelId for model in models]
    
    except Exception as e:
        print(f"❌ Error searching HF Hub: {e}")
        return []