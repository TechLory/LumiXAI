# Model Wrappers

Wrappers act as translation layers between external AI libraries and the LumiXAI framework. They handle downloading weights, initializing tokenizers, and routing inputs to the correct hardware device (CUDA, MPS, or CPU).

## Text Generation
Handles autoregressive language models (e.g., GPT-2, LLaMA, Qwen). It manages the sequential generation process and provides embeddings for gradient-based attribution.

::: src.wrappers.hf_text_generation.HFTextGenerationWrapper
    options:
      show_root_heading: false

## Text Classification
Handles sequence classification models (e.g., BERT, DistilBERT). It manages padding, masking, and returns logits for the predicted or target classes.

::: src.wrappers.hf_text_classification.HFTextClassificationWrapper
    options:
      show_root_heading: false

## Image Generation
Handles Latent Diffusion Models (e.g., Stable Diffusion) via the Hugging Face Diffusers library.

::: src.wrappers.hf_image.HFImageWrapper
    options:
      show_root_heading: false