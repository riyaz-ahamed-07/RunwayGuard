"""Hugging Face Space entrypoint."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_WEIGHTS_REPO", "DarkKnight1217/RunwayGuard-rtdetr-l")
os.environ.setdefault("RUNWAYGUARD_LLM", "0")

from deploy.gradio_app import demo  # noqa: E402

demo.queue(default_concurrency_limit=1).launch()
