import os
from functools import lru_cache
from pathlib import Path

from huggingface_hub import snapshot_download


@lru_cache(maxsize=1)
def get_model_root() -> Path:
    repo_id = os.environ.get(
        "MODEL_REPO_ID", "addy2002/unified-ml-models"
    )

    token = os.environ.get("HF_TOKEN") or True

    try:
        path = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            token=token,
            max_workers=4,
        )
    except Exception as error:
        raise RuntimeError(
            f"Could not download private model repository '{repo_id}'. "
            "Check HF_TOKEN, Hugging Face login, and repository permissions."
        ) from error

    return Path(path)