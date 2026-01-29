# Modular XAI Framework

## Overview
A modular full-stack framework designed to unify and simplify the Explainable AI (XAI) workflow. Its extensible architecture provides a standardized environment to interpret diverse model sources (such as Hugging Face) and integrate various attribution methods, serving as a flexible bridge between complex AI models and human interpretability.

* **Backend**: FastAPI, PyTorch, Transformers, Captum.
* **Frontend**: Next.js (React), Tailwind CSS.

## Tests
#### (gpt2-large + captum integrated gradients)
![alt text](/readme_files/gpt2-large_captum_ig_1.png)
#### (distilbert-base-uncased-finetuned-sst-2-english + captum integrated gradients) - NEGATIVE
![alt text](/readme_files/distilbert-base-uncased-finetuned-sst-2-english_captum_ig_negative.png)
#### (distilbert-base-uncased-finetuned-sst-2-english + captum integrated gradients) - POSITIVE
![alt text](/readme_files/distilbert-base-uncased-finetuned-sst-2-english_captum_ig_positive.png)
#### (OFA-Sys/small-stable-diffusion-v0 + DAAM) - STRONG ATTENTION SINK!!
![alt text](/readme_files/test-stable-diffusion-attention-sink.jpg)

## Quick Start

### Prerequisites
* Python 3.10+ & Poetry
* Node.js 18+ & npm

### Installation & Running

**1. Backend (Port 8000)**
```bash
cd backend
poetry install
poetry run uvicorn main:app --reload
```

**2. Frontend (Port 3000)**
```bash
cd frontend
npm install
npm run dev
```
Access the UI at [http://localhost:3000](http://localhost:3000).

## Extensibility
The framework uses a **Registry Pattern** in `backend/main.py` for easy expansion.

* **Add Models**: Create a class inheriting from `BaseWrapper` in `src/xai_framework/wrappers/` and register it in `AVAILABLE_WRAPPERS`.
* **Add XAI Methods**: Create a class inheriting from `BaseAttributor` in `src/xai_framework/attributors/` and register it in `AVAILABLE_ATTRIBUTORS`.


## License
This project is developed as part of a research thesis at the **University of Milan (Unimi)**.

## Contact
For any issues, bugs, or questions regarding this framework, please contact me at lorenzo.gatta@studenti.unimi.it or open an issue on GitHub.