# Base Classes

The core of the LumiXAI framework relies on abstract base classes. These interfaces ensure that any new model wrapper or attribution algorithm seamlessly integrates with the rest of the application, including the API layer and the frontend.

## BaseWrapper

The `BaseWrapper` class is responsible for loading the model weights and exposing a standardized inference method. By inheriting from this class, developers can wrap arbitrary models (e.g., from Hugging Face or custom local checkpoints) while hiding the underlying library-specific complexity.

::: src.abstract.BaseWrapper
    options:
      show_root_heading: false
      show_root_toc_entry: false

## BaseAttributor

The `BaseAttributor` class defines the interface for all Explainable AI methods. It takes an instantiated wrapper and computes the feature importance, returning a standardized `AttributionOutput` object that the Next.js frontend can visually render.

::: src.abstract.BaseAttributor
    options:
      show_root_heading: false
      show_root_toc_entry: false