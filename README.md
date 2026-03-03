# LumiXAI: Modular & Interactive Explainable AI Framework

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![React Next.js](https://img.shields.io/badge/frontend-Next.js-black.svg)
![PyTorch](https://img.shields.io/badge/backend-PyTorch-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview
**LumiXAI** is a modular, full-stack framework designed to unify and simplify the Explainable AI (XAI) workflow. 

Moving beyond static heatmaps, this framework provides a **highly interactive, bidirectional environment** to interpret generative models (Text-to-Text and Text-to-Image). Its extensible architecture acts as a standardized bridge between complex AI models (e.g., Hugging Face Hub) and human interpretability.

### Key Features
* **Bidirectional Text Generation Analysis:** Click on any generated output token to see which input tokens influenced it, or click an input token to see its causal effect on future generated text.
* **Interactive Pixel-Level Attribution:** Hover over high-resolution generated images to instantly inspect the textual attention (DAAM) driving specific spatial regions, directly mapping pixels back to prompt tokens.
* **Plug-and-Play Architecture:** Built with a Registry Pattern, allowing researchers to seamlessly add new models or attribution algorithms.
* **Hardware Auto-Detection:** Smart fallback routing across CUDA (NVIDIA), MPS (Apple Silicon), and CPU.

---

## Showcase & Demos

### 1. Autoregressive Text Generation (Captum IG)
Interactive matrix to track the context flow during text generation.
[Demo Text Generation](/readme_files/demo_text-to-text.gif)

### 2. Spatial Image Attribution (DAAM)
Real-time hovering to inspect stable diffusion attention maps.
[Demo Image Generation](/readme_files/demo_1_token-pixel_attribution_gui.gif)

<details>
<summary><b>View Static Attribution Tests (Click to expand)</b></summary>

* **Text Classification (DistilBERT):**
  * Negative Sentiment: [Show Image](/readme_files/distilbert-base-uncased-finetuned-sst-2-english_captum_ig_negative.png)
  * Positive Sentiment: [Show Image](/readme_files/distilbert-base-uncased-finetuned-sst-2-english_captum_ig_positive.png)
* **Image Generation Analysis (DAAM Attention Sinks):**
  * Baseline Sinks: [Show Image](/readme_files/test-stable-diffusion-attention-sink.jpg)
  * Special Tokens Removed: [Show Image](/readme_files/test-stable-diffusion-attention-sink-2-special-tokens-removed.png)
  * Percentile Norm (Dreamlike Diffusion): [Show Image](/readme_files/test-stable-diffusion-attention-sink-3-new-normalization.png)
</details>

---

## Quick Start

### Prerequisites
* Python 3.10+ & Poetry
* Node.js 18+ & npm

### Installation & Running

**1. Backend (FastAPI)**
```bash
cd backend
poetry install
poetry run uvicorn main:app --reload
```
*The backend runs on port 8000. API Documentation (Swagger) is auto-generated at `http://localhost:8000/docs`.*

**2. Frontend (Next.js)**
```bash
cd frontend
npm install
npm run dev
```
Access the interactive UI at [http://localhost:3000](http://localhost:3000).

---

## Extensibility & Architecture
The framework uses a strict **Registry Pattern** in `backend/main.py` for effortless expansion.

* **Add Models**: Create a class inheriting from `BaseWrapper` in `src/xai_framework/wrappers/` and register it in `AVAILABLE_WRAPPERS`.
* **Add XAI Methods**: Create a class inheriting from `BaseAttributor` in `src/xai_framework/attributors/` and register it in `AVAILABLE_ATTRIBUTORS`.

*(Note: Full developer documentation on how to write custom wrappers is coming soon in the `docs/` folder).*

---

## About & Contact
This project is developed as part of a research thesis at the **University of Milan (Unimi)**.

For any issues, bugs, or questions regarding this framework, please contact me at **lorenzo.gatta@studenti.unimi.it** or open an issue on this repository.

---



DOCUMENTAZIONE LOCALE (NO REMOTO)
poetry run mkdocs serve -a 127.0.0.1:8001