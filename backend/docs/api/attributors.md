# Attribution Algorithms

The `attributors` module contains the concrete implementations of Explainable AI algorithms. These classes inherit from `BaseAttributor` and are responsible for deeply inspecting the active model wrapper to extract feature importance scores (e.g., gradients or attention weights).

## Captum Integrated Gradients

Captum's Integrated Gradients, DeepLift, Saliency, Input x Gradient, GradientSHAP, Occlusion, and LIME implementations in this module are all "universal" attributors: they dynamically switch between a standard sequence classification pipeline, a step-by-step autoregressive generation pipeline, and a pixel-space image classification pipeline, based on the type of `BaseWrapper` injected. For text, they attribute to token embeddings; for image classification (`HFImageClassificationWrapper`), they attribute directly to the normalized pixel tensor, since pixel values are already continuous and need no embedding indirection.

::: src.attributors.captum_grad.CaptumGradientsAttributor
    options:
      show_root_heading: false
      show_root_toc_entry: false

## SmoothGrad

SmoothGrad averages Saliency over multiple noisy copies of an input image (via Captum's `NoiseTunnel`), cleaning up the salt-and-pepper look of raw single-pass pixel gradients. It is image-classification-only and does not carry the universal text dispatch of the attributors above.

::: src.attributors.captum_smoothgrad.CaptumSmoothGradAttributor
    options:
      show_root_heading: false
      show_root_toc_entry: false

## Grad-CAM

Grad-CAM (via Captum's `LayerGradCam`) attributes at a deep spatial feature map rather than raw pixel space, producing smoother, object-following heatmaps. The target layer is resolved generically at runtime, so it works for both CNNs and ViTs without per-model configuration. Like SmoothGrad, it is image-classification-only.

::: src.attributors.captum_gradcam.CaptumGradCamAttributor
    options:
      show_root_heading: false
      show_root_toc_entry: false

## Diffusion Attentive Attribution Maps (DAAM)

DAAM is specialized for Text-to-Image diffusion models. It intercepts cross-attention maps during the denoising process, allowing the framework to attribute specific pixel regions to exact prompt tokens.

::: src.attributors.daam.DAAMAttributor
    options:
      show_root_heading: false
      show_root_toc_entry: false