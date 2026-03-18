# FastAPI Endpoints

The `main.py` module is the entrypoint of the LumiXAI Backend. It exposes the REST API, manages the asynchronous background task queue for heavy inferences, and safeguards the hardware using global thread locks (`gpu_lock`) to prevent concurrent VRAM overloads.

::: main
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false