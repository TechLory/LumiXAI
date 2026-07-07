# Attribution Algorithms

The `attributors` module contains the concrete implementations of Explainable AI algorithms. These classes inherit from `BaseAttributor` and are responsible for deeply inspecting the active model wrapper to extract feature importance scores (e.g., gradients or attention weights).

## Captum Integrated Gradients

Captum's Integrated Gradients, DeepLift, Saliency, Input x Gradient, GradientSHAP, Occlusion, and LIME implementations in this module are all "universal" attributors: they dynamically switch between a standard sequence classification pipeline, a step-by-step autoregressive generation pipeline, and a pixel-space image classification pipeline, based on the type of `BaseWrapper` injected. For text, they attribute to token embeddings; for image classification (`HFImageClassificationWrapper`), they attribute directly to the normalized pixel tensor, since pixel values are already continuous and need no embedding indirection.

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