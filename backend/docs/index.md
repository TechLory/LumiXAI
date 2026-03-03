# Welcome to LumiXAI 🌟

**LumiXAI** is a modular, full-stack Explainable AI (XAI) framework designed to bridge the gap between complex generative models and human interpretability.

Unlike traditional tools that provide static heatmaps, LumiXAI offers a **highly interactive, bidirectional environment** to analyze both Text-to-Text and Text-to-Image models.

## Key Capabilities

* **Bidirectional Text Generation Analysis:** Inspect autoregressive generation step-by-step. Click generated tokens to trace back their causal inputs, or select input tokens to see their forward influence.
* **Spatial Image Attribution:** Hover over high-resolution generated images to instantly reveal the latent textual attention (DAAM) driving specific regions.
* **Extensible Architecture:** Built on a robust Registry Pattern, making it trivial to plug in new Hugging Face models or custom XAI algorithms without altering the core engine.

## Navigating the Documentation

Use the top navigation bar to explore the framework's architecture:

* **API Reference**: Detailed documentation of the core classes, including `BaseWrapper` and `BaseAttributor`, which serve as the foundation for extending the framework.
* **Wrappers**: Implementations that connect external libraries (like Hugging Face Transformers and Diffusers) to the LumiXAI ecosystem.
* **Attributors**: The mathematical and algorithmic core where XAI methods like Captum Integrated Gradients and DAAM are implemented.