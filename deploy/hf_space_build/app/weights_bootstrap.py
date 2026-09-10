"""Ensure MODEL_PATH exists; download from HF_WEIGHTS_REPO if missing."""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_WEIGHTS_REPO = "DarkKnight1217/RunwayGuard-rtdetr-l"


def ensure_weights() -> Path:
    target = Path(os.environ.get("MODEL_PATH", "weights/best.pt"))
    if target.is_file():
        os.environ["MODEL_PATH"] = str(target)
        return target

    repo = os.environ.get("HF_WEIGHTS_REPO", DEFAULT_WEIGHTS_REPO).strip()
    if not repo:
        raise FileNotFoundError(
            f"Missing weights at {target}. Set MODEL_PATH or HF_WEIGHTS_REPO."
        )

    from huggingface_hub import hf_hub_download

    downloaded = Path(hf_hub_download(repo_id=repo, filename="best.pt"))
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        try:
            target.write_bytes(downloaded.read_bytes())
        except OSError:
            os.environ["MODEL_PATH"] = str(downloaded)
            return downloaded
    os.environ["MODEL_PATH"] = str(target)
    return target
