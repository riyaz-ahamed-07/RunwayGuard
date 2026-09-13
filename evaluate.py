from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import RTDETR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RunwayGuard on its untouched grouped test split.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=480)
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
    names = getattr(metrics, "names", None) or getattr(model, "names", {})
    per_class = [float(value) for value in box.maps]
    result = {
        "map50_95": float(box.map),
        "map50": float(box.map50),
        "map75": float(box.map75),
        "precision": _optional_float(getattr(box, "mp", None)),
        "recall": _optional_float(getattr(box, "mr", None)),
        "per_class_map50_95": per_class,
        "per_class_names": [str(names.get(i, i)) for i in range(len(per_class))]
        if isinstance(names, dict)
        else [str(i) for i in range(len(per_class))],
        "fitness": float(metrics.fitness),
        "weights": str(args.weights),
        "data": str(args.data),
        "imgsz": args.imgsz,
        "split": "test",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
