<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="readme_files/logo-darkmode.svg">
    <source media="(prefers-color-scheme: light)" srcset="readme_files/logo-lightmode.svg">
    <img alt="LumiXAI" src="readme_files/logo-lightmode.svg" width="520">
  </picture>
</p>

# LumiXAI: Modular & Interactive Explainable AI Framework

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![React Next.js](https://img.shields.io/badge/frontend-Next.js-black.svg)
![PyTorch](https://img.shields.io/badge/backend-PyTorch-ee4c2c.svg)
![Documentation](https://img.shields.io/badge/docs-MkDocs-083e8c.svg)
![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-green.svg)
![No AI Training](https://img.shields.io/badge/No%20AI%20Training-Prohibited-red)

## Table of Contents
- [Overview](#overview)
  - [Key Features](#key-features)
  - [Currently Supported Models](#currently-supported-models)
- [Getting Started](#getting-started)
  - [Option A: Running with Docker](#option-a-running-with-docker-recommended-for-linuxwindows-servers)
  - [Option B: Local Installation via Poetry](#option-b-local-installation-via-poetry-recommended-for-macos)
- [Guided Tutorials](#guided-tutorials)
- [Case Study Notebooks](#case-study-notebooks)
- [Sharing One Backend](#sharing-one-backend)
- [Extensibility & Architecture](#extensibility--architecture)
- [About & Contact](#about--contact)
  - [AI/ML Training Opt-Out](#aiml-training-opt-out)

## Overview
**LumiXAI** is a modular, full-stack framework designed to unify and simplify the Explainable AI (XAI) workflow. 

Moving beyond static heatmaps, this framework provides a **highly interactive, bidirectional environment** to interpret generative models (Text-to-Text and Text-to-Image). Its extensible architecture acts as a standardized bridge between complex AI models (e.g., Hugging Face Hub) and human interpretability.

### Key Features
* **Bidirectional Text Generation Analysis:** Click on any generated output token to see which input tokens influenced it, or click an input token to see its causal effect on future generated text.
* **Interactive Pixel-Level Attribution:** Hover over high-resolution generated images to instantly inspect the textual attention (DAAM) driving specific spatial regions, directly mapping pixels back to prompt tokens.
* **9 Attribution Methods:** Integrated Gradients, DeepLift, Saliency, Input x Gradient, GradientSHAP, Occlusion, and LIME (SLIC superpixels) run universally across text and image classification models; SmoothGrad and Grad-CAM are image-classification-only, plus DAAM for diffusion image generation.
* **Plug-and-Play Architecture:** Built with a Registry Pattern, **allowing researchers to seamlessly add new models or attribution algorithms**.
* **Hardware Auto-Detection:** Smart fallback routing across CUDA (NVIDIA), MPS (Apple Silicon), and CPU.

### Currently Supported Models
LumiXAI is designed with a dynamic routing heuristic that provides out-of-the-box support for the most widely used model families on the Hugging Face Hub:

* **Standard Image Generation:** Stable Diffusion 1.5, 2.1, and SDXL 1.0 (defaults to 30 inference steps with CFG enabled).
* **Fast/Distilled Image Generation:** SD-Turbo (1-step) and SDXL-Turbo (4-steps) with dynamic CFG disabling to prevent inference crashes.
* **Image Classification:** ViT, ResNet, ConvNeXt, and compatible architectures, with attribution rendered against the model's own preprocessed (de-normalized) pixel tensor for pixel-perfect overlays.
* **Text Classification:** BERT, DistilBERT, RoBERTa, and compatible transformer architectures, with normalized percentage confidence scores per token.
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

For machine-specific settings that should not be committed, create a repo-root `.env.local` file and start Compose with it:
```bash
cp .env.local.example .env.local
docker compose --env-file .env.local -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```
- `.env.local` is ignored by git and is the right place for custom ports, `HF_TOKEN`, `LUMIXAI_DEFAULT_DEVICE`, and `LUMIXAI_VISIBLE_GPUS`.
- The example file includes `LUMIXAI_FRONTEND_PORT`, `LUMIXAI_BACKEND_PORT`, and `LUMIXAI_DOCS_PORT` if you want host-side port overrides.
- You can also set `LUMIXAI_TEXT_MAX_NEW_TOKENS` there to raise the text generation cap while still allowing EOS to stop generation earlier.
- `LUMIXAI_MODEL_IDLE_TIMEOUT_SEC` (default `1800`) controls how long a loaded model may sit unused before the backend unloads it and frees the (V)RAM. The timer resets on every load and explanation, and never fires while a job is running. Set it to `0` to keep models resident until they are unloaded by hand.

To pin the backend to a specific GPU, set one or both of these environment variables before starting Compose:
```bash
export LUMIXAI_DEFAULT_DEVICE=cuda:1
export LUMIXAI_VISIBLE_GPUS=1
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```
- `LUMIXAI_DEFAULT_DEVICE` controls which device the backend loads models onto. Supported values include `auto`, `cpu`, `cuda`, `cuda:0`, `cuda:1`, and `mps`.
- `LUMIXAI_VISIBLE_GPUS` limits which physical NVIDIA GPU IDs are exposed to the container. If you set `LUMIXAI_VISIBLE_GPUS=1`, that GPU becomes the container's visible CUDA device `0`.

To use gated or private Hugging Face repos, first request access on the model page and create a read token on Hugging Face. Then export it before starting Compose:
```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```
- The backend forwards `HF_TOKEN` to Hugging Face Hub API calls and `from_pretrained(...)` downloads.
- Without `HF_TOKEN`, LumiXAI hides gated/private repos from search and rejects them at load time with a friendly error.
- With `HF_TOKEN`, gated/private repos can be searched and loaded as long as that token's account actually has access.

**To stop the application:**
```bash
docker compose down
```




**Services Architecture:**
- Frontend Interface: `http://localhost:3000`
- API Backend: `http://localhost:8000`
- Documentation: `http://localhost:8001`

*(Note: Downloaded HF models and SQLite database states are preserved locally via bind mounts).*

By default the frontend talks to the API on port 8000 of the host it was loaded from (it builds the URL from the browser's hostname). The documentation is served at `/docs` on the **same origin** — i.e. through the reverse proxy in a deployed setup (`http://<host>/docs`); when the app is opened on a local dev port (3000/3001) it instead loads the docs from port 8001 of the same host. Either can be overridden via `frontend/.env.local` — useful when the frontend is accessed from a different machine than the one running the backend/docs:
```
NEXT_PUBLIC_API_BASE_URL=http://<backend-host>:8000
NEXT_PUBLIC_DOCS_BASE_URL=http://<backend-host>:8001
```
Restart the frontend after changing this file for it to take effect.

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

## Guided Tutorials
From the welcome screen, LumiXAI offers four self-contained, interactive walkthroughs — no model loading or backend inference required, since each one replays a real, pre-computed result:

* **Text Classification:** See which words drove a real sentiment prediction (CardiffNLP Twitter RoBERTa).
* **Text Generation:** Trace a real reply back to the prompt that shaped it (Qwen3-0.6B).
* **Text to Image:** Map prompt words onto the pixels they generated (SDXL).
* **Image Classification:** See which pixels drove a real MNIST digit prediction (ViT), via a Grad-CAM overlay.

Each tutorial walks step-by-step through the full workflow — choosing a source, model, and attributor, loading the configuration, providing input, and running attribution — then lets you interact with the resulting heatmap: click input tokens to see their forward influence, click output tokens/regions to trace them back, and find the run pinned in Job History.

---

## Case Study Notebooks
Beyond the guided tutorials, [`backend/notebooks/`](backend/notebooks) contains research-oriented Jupyter notebooks that drive the running backend through the shipped [`client.py`](backend/notebooks/client.py) via `run_smart_batch`, submitting real batches of jobs and exploring the results interactively in the frontend's Job History. Each one replicates a corresponding experiment under [`experiments/`](experiments):

* **[`case_study_bold_generation.ipynb`](backend/notebooks/case_study_bold_generation.ipynb)** — a curated slice of the [BOLD](https://arxiv.org/abs/2101.11718) benchmark run deterministically through `gpt2` with `captum_ig`, so a generated word's attribution back to its prompt/context tokens can be inspected click-by-click.
* **[`case_study_diffusion_style.ipynb`](backend/notebooks/case_study_diffusion_style.ipynb)** — a `content x style` prompt grid (e.g. `cow` vs. `Rembrandt`) run through SDXL with the `daam` attributor, comparing per-token attention heatmaps for content vs. style tokens on the same image.
* **[`case_study_toxicity_classification.ipynb`](backend/notebooks/case_study_toxicity_classification.ipynb)** — curated [Civil Comments](https://huggingface.co/datasets/google/civil_comments) rows through `unitary/toxic-bert`, attributing each row to its own toxicity subtype to check whether attribution lands on genuinely abusive language or over-attributes to identity terms.
* **[`case_study_1.ipynb`](backend/notebooks/case_study_1.ipynb)** and **[`case_study_2.ipynb`](backend/notebooks/case_study_2.ipynb)** — earlier examples covering IMDB sentiment classification bias and diffusion content/style attribution respectively.

**Prerequisites**: the LumiXAI backend running on `http://localhost:8000` (the diffusion notebooks additionally require GPU access).

---

## Sharing One Backend
The backend holds a **single model at a time**, while the frontend, notebooks and experiment scripts can all reach it at once. Two rules keep concurrent users from stepping on each other, and both surface as ordinary HTTP responses:

* **Results are pinned to a configuration.** Every load and attributor change mints a `config_id`, returned by `/api/load` and `/api/set_attributor`. Clients send it back on `/api/explain`, and a job whose configuration is no longer live is refused (`409`) rather than being answered by whatever model happens to be loaded now. Jobs are pinned at submission too, so a swap that happens while a job waits in the queue fails that job instead of mislabelling its result.
* **Whoever loads a model keeps it.** A client identifies itself with an `X-LumiXAI-Session` header and holds a lease on the configuration it loaded. Another client trying to load, change the attributor, or unload gets `423 Locked`, naming the model and when the claim lapses. The lease expires exactly when the idle reaper would release the model (see `LUMIXAI_MODEL_IDLE_TIMEOUT_SEC`), so nothing is ever locked forever. Send `"force": true` to take over deliberately — the frontend shows a **Take over and load anyway** button when this happens. Running attributions is never gated: clients share the loaded model freely, since that's the point of a shared box.

A configuration change is also refused (`409`) while any job is running, so weights are never freed out from under a computation in progress.

The shipped [`client.py`](backend/notebooks/client.py) does all of this for you — each `Client` instance is its own session. Pass `Client(base_url, force=True)` for a batch that should claim the backend regardless of who is using it. `GET /api/status` reports what is loaded, whether it is yours, and how long the lease and the model have left.

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
