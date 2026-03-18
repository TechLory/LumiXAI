# Model Wrappers

The `wrappers` module implements the Adapter Pattern to standardize interactions with various external machine learning libraries (primarily Hugging Face). By inheriting from `BaseWrapper`, these classes ensure that models expose uniform methods for loading, inference, and embedding extraction, regardless of their underlying architecture.

## Hugging Face: Text Generation
Wrapper for Auto-Regressive / Causal Language Models (e.g., GPT-2, LLaMA). Includes specialized methods to extract step-by-step probabilities for generation sequences.

::: src.wrappers.hf_text_generation.HFTextGenerationWrapper
    options:
      show_root_heading: false
      show_root_toc_entry: false

## Hugging Face: Text Classification
Wrapper for Sequence Classification Models (e.g., BERT, DistilBERT). Automates tokenization padding and extracts normalized prediction logits.

::: src.wrappers.hf_text_classification.HFTextClassificationWrapper
    options:
      show_root_heading: false
      show_root_toc_entry: false

## Hugging Face: Image Generation
Wrapper for Stable Diffusion and SDXL Diffusers. Manages automatic device offloading, precision casting (fp16 vs fp32), and dynamic architecture detection.

::: src.wrappers.hf_image.HFImageWrapper
    options:
      show_root_heading: false
      show_root_toc_entry: false