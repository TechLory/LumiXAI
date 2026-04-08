# LumiXAI: Modular & Interactive Explainable AI Framework

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![React Next.js](https://img.shields.io/badge/frontend-Next.js-black.svg)
![PyTorch](https://img.shields.io/badge/backend-PyTorch-ee4c2c.svg)
![Documentation](https://img.shields.io/badge/docs-MkDocs-083e8c.svg)
![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-green.svg)
![No AI Training](https://img.shields.io/badge/No%20AI%20Training-Prohibited-red)

## Overview
**LumiXAI** is a modular, full-stack framework designed to unify and simplify the Explainable AI (XAI) workflow. 

Moving beyond static heatmaps, this framework provides a **highly interactive, bidirectional environment** to interpret generative models (Text-to-Text and Text-to-Image). Its extensible architecture acts as a standardized bridge between complex AI models (e.g., Hugging Face Hub) and human interpretability.

### Key Features
* **Bidirectional Text Generation Analysis:** Click on any generated output token to see which input tokens influenced it, or click an input token to see its causal effect on future generated text.
* **Interactive Pixel-Level Attribution:** Hover over high-resolution generated images to instantly inspect the textual attention (DAAM) driving specific spatial regions, directly mapping pixels back to prompt tokens.
* **Plug-and-Play Architecture:** Built with a Registry Pattern, allowing researchers to seamlessly add new models or attribution algorithms.
* **Hardware Auto-Detection:** Smart fallback routing across CUDA (NVIDIA), MPS (Apple Silicon), and CPU.

### Supported Models
LumiXAI is designed with a dynamic routing heuristic that provides out-of-the-box support for the most widely used model families on the Hugging Face Hub:

* **Standard Image Generation:** Stable Diffusion 1.5, 2.1, and SDXL 1.0 (defaults to 30 inference steps with CFG enabled).
* **Fast/Distilled Image Generation:** SD-Turbo (1-step) and SDXL-Turbo (4-steps) with dynamic CFG disabling to prevent inference crashes.
* **Text Classification:** BERT, DistilBERT, RoBERTa, and compatible transformer architectures.
* **Text Generation:** GPT-2 family and compatible autoregressive models.

*(Note: Models requiring highly specific generative hyperparameters, such as LCMs or Stable Diffusion 3, are not officially supported out-of-the-box but can be easily integrated by extending the wrapper logic in `src/attributors/daam.py`).*

---

## Getting Started

LumiXAI is designed to be highly portable. It can be executed via Docker for rapid deployment on servers (with NVIDIA GPU support) or installed locally for native hardware acceleration (recommended for Apple Silicon/MPS users).

### Option A: Running with Docker (Recommended for Linux/Windows Servers)

The framework includes a fully configured Docker ecosystem that launches the FastAPI backend, Next.js frontend, and MkDocs documentation simultaneously.

**Prerequisites:**
- Docker and Docker Compose.
- *Optional for GPU acceleration:* An NVIDIA GPU with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on the host machine.

**Execution:**
Navigate to the root directory of the project and run:

Without GPU support (CPU only):
```bash
docker compose up -d --build
```
With GPU (nvidia) support:
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```
*(Note: The GPU command will fail if the host machine does not have an NVIDIA GPU and the NVIDIA Container Toolkit installed..)*

To pin the backend to a specific GPU, set one or both of these environment variables before starting Compose:
```bash
export LUMIXAI_DEFAULT_DEVICE=cuda:1
export LUMIXAI_VISIBLE_GPUS=1
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```
- `LUMIXAI_DEFAULT_DEVICE` controls which device the backend loads models onto. Supported values include `auto`, `cpu`, `cuda`, `cuda:0`, `cuda:1`, and `mps`.
- `LUMIXAI_VISIBLE_GPUS` limits which physical NVIDIA GPU IDs are exposed to the container. If you set `LUMIXAI_VISIBLE_GPUS=1`, that GPU becomes the container's visible CUDA device `0`.

**To stop the application:**
```bash
docker compose down
```




**Services Architecture:**
- Frontend Interface: `http://localhost:3000`
- API Backend: `http://localhost:8000`
- Documentation: `http://localhost:8001`

*(Note: Downloaded HF models and SQLite database states are preserved locally via bind mounts).*

### Option B: Local Installation via Poetry (Recommended for macOS)

For macOS users (Intel or Apple Silicon) or developers who wish to utilize native hardware acceleration (MPS) not supported by Docker passthrough, a local setup is required.

**Prerequisites:**
- Python 3.11 or 3.12
- [Poetry](https://python-poetry.org/) package manager
- Node.js 18+

**1. Backend Initialization:**
```bash
cd backend
poetry install
poetry run uvicorn main:app --host 0.0.0.0 --port 8000
```
*The system will automatically detect and utilize available hardware accelerators (CUDA, MPS, or fallback to CPU).*

**2. Frontend Initialization:**
In a new terminal instance:
```bash
cd frontend
npm install
npm run dev
```

**3. Documentation Initialization:**
In a new terminal instance:
```bash
cd backend
poetry run mkdocs serve -a 127.0.0.1:8001
```

---

## Showcase & Demos
1. Autoregressive Text Generation (Captum IG): [GIF](/readme_files/demo_text-to-text.gif)
2. Spatial Image Attribution (DAAM): [GIF](/readme_files/demo_1_token-pixel_attribution_gui.gif)

More demos coming soon...
<!-- <details>
<summary><b>View Static Attribution Tests (Click to expand)</b></summary>

* **Text Classification (DistilBERT):**
  * Negative Sentiment: [Show Image](/readme_files/distilbert-base-uncased-finetuned-sst-2-english_captum_ig_negative.png)
  * Positive Sentiment: [Show Image](/readme_files/distilbert-base-uncased-finetuned-sst-2-english_captum_ig_positive.png)
* **Image Generation Analysis (DAAM Attention Sinks):**
  * Baseline Sinks: [Show Image](/readme_files/test-stable-diffusion-attention-sink.jpg)
  * Special Tokens Removed: [Show Image](/readme_files/test-stable-diffusion-attention-sink-2-special-tokens-removed.png)
  * Percentile Norm (Dreamlike Diffusion): [Show Image](/readme_files/test-stable-diffusion-attention-sink-3-new-normalization.png)
</details> -->



---

## Extensibility & Architecture
The framework uses a strict **Registry Pattern** in `backend/main.py` for effortless expansion.

* **Add Models**: Create a class inheriting from `BaseWrapper` in `src/wrappers/` and register it in `AVAILABLE_WRAPPERS`.
* **Add XAI Methods**: Create a class inheriting from `BaseAttributor` in `src/attributors/` and register it in `AVAILABLE_ATTRIBUTORS`.

*(Note: Full developer documentation on how to write custom wrappers is coming soon in the `docs/` folder).*

---

## About & Contact
This project is developed as part of a research thesis at the **University of Milan (Unimi)**.

For any issues, bugs, or questions regarding this framework, please contact me at **lorenzo.gatta@studenti.unimi.it** or open an issue on this repository.

---

### AI/ML Training Opt-Out
The author(s) of this repository explicitly prohibit the use of the code, data, and content within this project for the training, fine-tuning, or development of machine learning models, artificial intelligence algorithms, or large language models (LLMs). This includes scraping, text and data mining (TDM), and any commercial or non-commercial use by AI entities.
