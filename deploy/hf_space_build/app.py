"""Hugging Face Space entrypoint."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_WEIGHTS_REPO", "DarkKnight1217/RunwayGuard-rtdetr-l")
os.environ.setdefault("RUNWAYGUARD_LLM", "0")

# Gradio client can crash on boolean JSON-schema additionalProperties (pydantic 2.11+).
try:
    import gradio_client.utils as _gcu

    _orig_get_type = _gcu.get_type

    def _get_type(schema):  # type: ignore[no-untyped-def]
        if isinstance(schema, bool):
            return "Any"
        return _orig_get_type(schema)

    _gcu.get_type = _get_type
except Exception:
    pass

from deploy.gradio_app import SAMPLES_DIR, demo  # noqa: E402

demo.queue(default_concurrency_limit=1).launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", "7860")),
    allowed_paths=[str(SAMPLES_DIR.resolve())],
)
