# Utility Modules

The utilities folder contains specialized helper functions and highly customized logic that supports the core framework infrastructure, such as external API integrations and low-level tensor tracing.

## Hugging Face Hub

Provides lightweight search capabilities connecting directly to the Hugging Face API to populate the frontend model registry.

::: src.utils.hf_hub
    options:
      show_root_heading: false
      show_root_toc_entry: false

## Custom DAAM Core

A highly customized re-implementation of the Diffusion Attentive Attribution Maps tracing logic. It intercepts PyTorch Cross-Attention layers (`CaptureAttnProcessor`) dynamically during the generation loop.

::: src.utils.daam_custom
    options:
      show_root_heading: false
      show_root_toc_entry: false