# XAI Attributors

Attributors contain the algorithmic logic to interpret the model's decisions. They query the wrapper's internal state (gradients, attention maps) and format the results for interactive visualization.

## Captum Integrated Gradients
Used primarily for NLP tasks (Text Generation and Classification). It computes the integral of gradients with respect to the input embeddings along a straight path from a baseline to the actual input. For autoregressive generation, this is applied iteratively per generated token.

::: src.attributors.captum_grad.CaptumGradientsAttributor
    options:
      show_root_heading: false

## Diffusion Attentive Attribution Maps (DAAM)
Designed specifically for Text-to-Image diffusion models. It intercepts and aggregates the cross-attention maps during the denoising steps to create a spatial heatmap linking specific prompt words to regions in the generated image.

::: src.attributors.daam.DAAMAttributor
    options:
      show_root_heading: false