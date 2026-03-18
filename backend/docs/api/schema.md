# Core Schemas

The schema module defines the foundational data structures of the LumiXAI framework using Pydantic. These models serve as the strict data contract between the backend attribution algorithms and the frontend visualization layer, ensuring consistency and type safety across the entire pipeline.

## InputFeature

The `InputFeature` class provides a modality-agnostic representation of the data being analyzed. Whether the input is a text token from a Large Language Model or a spatial patch from a Diffusion model, this schema standardizes its properties.

::: src.schema.InputFeature
    options:
      show_root_heading: false
      show_root_toc_entry: false

## AttributionOutput

The `AttributionOutput` class is the mandatory return type for any concrete implementation of the `BaseAttributor`. It bundles the raw attribution matrices (heatmaps) with their corresponding input features and metadata, ready to be serialized and consumed by the interactive GUI.

::: src.schema.AttributionOutput
    options:
      show_root_heading: false
      show_root_toc_entry: false