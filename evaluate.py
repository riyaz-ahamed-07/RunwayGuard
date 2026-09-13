from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import RTDETR

IMGSZ_DEFAULT = 480
ASK_EVIDENCE_CONFIDENCE = 0.25
ASK_ANSWER_CONFIDENCE = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RunwayGuard on its untouched grouped test split.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=IMGSZ_DEFAULT)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, default=Path("test_metrics.json"))
    return parser.parse_args()


def _optional_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def main() -> None:
    args = parse_args()
    model = RTDETR(str(args.weights))
    metrics = model.val(
        data=str(args.data),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        plots=True,
        verbose=True,
    )
    box = metrics.box
    names = getattr(metrics, "names", None) or {}
    per_class = [float(value) for value in box.maps]
    result = {
        "note": (
            "map/precision/recall below follow Ultralytics' COCO-style aggregation. "
            "They are not automatically equal to API operating points. "
            f"Documented API thresholds: evidence={ASK_EVIDENCE_CONFIDENCE}, "
            f"answer={ASK_ANSWER_CONFIDENCE}, imgsz={IMGSZ}."
        ),
        "map50_95": float(box.map),
        "map50": float(box.map50),
        "map75": float(box.map75),
        "ultralytics_mean_precision": _optional_float(getattr(box, "mp", None)),
        "ultralytics_mean_recall": _optional_float(getattr(box, "mr", None)),
        "per_class_map50_95": per_class,
        "per_class_names": [str(names.get(i, i)) for i in range(len(per_class))]
        if isinstance(names, dict)
        else [str(i) for i in range(len(per_class))],
        "fitness": float(metrics.fitness),
        "weights": str(args.weights),
        "data": str(args.data),
        "imgsz": args.imgsz,
        "split": "test",
        "api_evidence_confidence": ASK_EVIDENCE_CONFIDENCE,
        "api_answer_confidence": ASK_ANSWER_CONFIDENCE,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
