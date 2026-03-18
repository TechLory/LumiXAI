# Attribution Algorithms

The `attributors` module contains the concrete implementations of Explainable AI algorithms. These classes inherit from `BaseAttributor` and are responsible for deeply inspecting the active model wrapper to extract feature importance scores (e.g., gradients or attention weights).

## Captum Integrated Gradients

Captum is used for text-based modalities. This attributor dynamically switches between a standard sequence classification pipeline and a complex, step-by-step autoregressive generation pipeline based on the model loaded.

::: src.attributors.captum_grad.CaptumGradientsAttributor
    options:
      show_root_heading: false
      show_root_toc_entry: false

## Diffusion Attentive Attribution Maps (DAAM)

DAAM is specialized for Text-to-Image diffusion models. It intercepts cross-attention maps during the denoising process, allowing the framework to attribute specific pixel regions to exact prompt tokens.

::: src.attributors.daam.DAAMAttributor
    options:
      show_root_heading: false
      show_root_toc_entry: false