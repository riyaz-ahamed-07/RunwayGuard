from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TAXONOMY_PATH = ROOT / "config" / "taxonomy.json"

# Unified train/eval/serve resolution (must match evaluate.py / train defaults).
IMGSZ = 480

# /detect form default: display / NMS-style score filter only.
DETECT_DISPLAY_CONFIDENCE = 0.25

# /ask always collects evidence at this fixed server threshold (callers cannot hide mid-scores).
ASK_EVIDENCE_CONFIDENCE = 0.25

# Part B may answer only from detections at or above this score.
ASK_ANSWER_CONFIDENCE = 0.50

CLASS_NAMES: tuple[str, ...] = tuple(json.loads(TAXONOMY_PATH.read_text(encoding="utf-8")).keys())
