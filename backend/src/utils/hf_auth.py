import os
from typing import Dict

HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def get_hf_token() -> str | None:
    for env_var in HF_TOKEN_ENV_VARS:
        token = os.getenv(env_var, "").strip()
        if token:
            return token
    return None


def has_hf_token() -> bool:
    return get_hf_token() is not None


def hf_auth_kwargs() -> Dict[str, str]:
    token = get_hf_token()
    return {"token": token} if token else {}
