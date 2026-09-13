from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import torch
import ultralytics
from ultralytics import RTDETR

IMGSZ_DEFAULT = 480


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune RT-DETR on RunwayGuard FOD data.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="rtdetr-l.pt")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=IMGSZ_DEFAULT)
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
        help="Resume from project/name/weights/last.pt (fails if missing).",
    )
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    resume = bool(args.resume)
    if resume and not last_ckpt.exists():
        raise FileNotFoundError(
            f"--resume was set but checkpoint is missing: {last_ckpt}. "
            "Start a new run without --resume, or point --project/--name at an existing attempt."
        )
    checkpoint = str(last_ckpt) if resume else args.model
    started = time.time()

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": getattr(__import__("torchvision", fromlist=["__version__"]), "__version__", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "ultralytics": ultralytics.__version__,
        "resume": resume,
        "checkpoint": checkpoint,
        "checkpoint_sha256": _sha256(Path(checkpoint)) if Path(checkpoint).exists() else None,
        "data": str(args.data),
        "data_yaml_sha256": _sha256(Path(args.data)),
        "arguments": vars(args) | {"data": str(args.data), "project": str(args.project)},
        "started_unix": started,
    }
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (run_dir / f"environment_{stamp}.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
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
        # RT-DETR deformable attention lacks a fully deterministic CUDA backward.
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
    environment["elapsed_seconds"] = round(time.time() - started, 1)
    (run_dir / f"environment_{stamp}_complete.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
