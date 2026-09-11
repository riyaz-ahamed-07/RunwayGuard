from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch
import ultralytics
from ultralytics import RTDETR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune RT-DETR on RunwayGuard FOD data.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="rtdetr-l.pt")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=Path, default=Path("runs"))
    parser.add_argument("--name", default="rtdetr_l_foda")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from project/name/weights/last.pt if present.",
    )
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.epochs = 1
        args.batch = min(args.batch, 4)
        args.fraction = min(args.fraction, 0.02)
        args.name = f"{args.name}_smoke"

    run_dir = args.project / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    last_ckpt = run_dir / "weights" / "last.pt"
    resume = bool(args.resume and last_ckpt.exists())
    checkpoint = str(last_ckpt) if resume else args.model

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "ultralytics": ultralytics.__version__,
        "resume": resume,
        "checkpoint": checkpoint,
        "arguments": vars(args) | {"data": str(args.data), "project": str(args.project)},
    }
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    model = RTDETR(checkpoint)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        # RT-DETR deformable attention uses a CUDA operation without a fully
        # deterministic backward pass. The seed is still fixed and recorded.
        deterministic=False,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=1e-4,
        weight_decay=1e-4,
        warmup_epochs=min(2, max(args.epochs - 1, 0)),
        patience=args.patience,
        amp=True,
        plots=True,
        save=True,
        verbose=True,
        fraction=args.fraction,
        resume=resume,
    )


if __name__ == "__main__":
    main()
