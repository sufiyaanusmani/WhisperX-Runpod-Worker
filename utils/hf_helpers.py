import os

def get_huggingface_token() -> str:
    return os.environ.get("HF_TOKEN", "")