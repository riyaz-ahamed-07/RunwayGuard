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
    result = {
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "map75": float(metrics.box.map75),
        "per_class_map50_95": [float(value) for value in metrics.box.maps],
        "fitness": float(metrics.fitness),
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
