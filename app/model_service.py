from __future__ import annotations

import io
import logging
import os
import threading
import time
from pathlib import Path

from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError
from ultralytics import RTDETR

from .runtime_config import CLASS_NAMES, DETECT_DISPLAY_CONFIDENCE, IMGSZ
from .schemas import BoundingBox, DetectResponse, Detection, ImageSize


DEFAULT_CONFIDENCE = DETECT_DISPLAY_CONFIDENCE
LOGGER = logging.getLogger("runwayguard.inference")
_MODEL_LOCK = threading.RLock()
_MODEL: RTDETR | None = None
_MODEL_PATH: Path | None = None
MAX_DECODE_PIXELS = int(os.environ.get("MAX_DECODE_PIXELS", str(20_000_000)))


class InvalidImageError(ValueError):
    pass


def decode_image(content: bytes) -> Image.Image:
    try:
        probe = Image.open(io.BytesIO(content))
        width, height = probe.size
        if width <= 0 or height <= 0:
            raise InvalidImageError("The uploaded file has invalid image dimensions.")
        if width * height > MAX_DECODE_PIXELS:
            raise InvalidImageError(
                f"Image header reports {width * height} pixels, exceeding {MAX_DECODE_PIXELS}."
            )
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        image = ImageOps.exif_transpose(probe)
        image.load()
    except InvalidImageError:
        raise
    except Image.DecompressionBombError as exc:
        raise InvalidImageError("Image rejected as a decompression bomb.") from exc
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise InvalidImageError("The uploaded file is not a readable image.") from exc
    if image.width * image.height > MAX_DECODE_PIXELS:
        raise InvalidImageError(
            f"Decoded image exceeds {MAX_DECODE_PIXELS} pixels; resize before upload."
        )
    return image.convert("RGB")


def _resolved_model_path() -> Path:
    return Path(os.environ.get("MODEL_PATH", "weights/best.pt"))


def get_model() -> RTDETR:
    global _MODEL, _MODEL_PATH
    with _MODEL_LOCK:
        path = _resolved_model_path()
        if _MODEL is not None and _MODEL_PATH == path:
            return _MODEL
        if not path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {path}. Set MODEL_PATH to a local .pt file "
                "(download the checkpoint first; MODEL_PATH is not a URL)."
            )
        model = RTDETR(str(path))
        names = getattr(model, "names", None)
        if names is None and hasattr(model, "model"):
            names = getattr(model.model, "names", None)
        if isinstance(names, dict):
            ordered = [str(names[i]) for i in sorted(int(k) for k in names.keys())]
        elif names is not None:
            ordered = [str(name) for name in names]
        else:
            ordered = []
        if not ordered:
            raise RuntimeError(
                "Loaded checkpoint did not expose class names; refusing to serve with unknown mapping."
            )
        if tuple(ordered) != CLASS_NAMES:
            raise RuntimeError(
                "Loaded checkpoint class names do not match config/taxonomy.json order: "
                f"model={ordered} expected={list(CLASS_NAMES)}"
            )
        _MODEL = model
        _MODEL_PATH = path
        return _MODEL


def reset_model_cache() -> None:
    global _MODEL, _MODEL_PATH
    with _MODEL_LOCK:
        _MODEL = None
        _MODEL_PATH = None


def detect(image: Image.Image, confidence: float = DEFAULT_CONFIDENCE) -> DetectResponse:
    with _MODEL_LOCK:
        model = get_model()
        started = time.perf_counter()
        try:
            import torch

            device = 0 if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            device = "cpu"
        result = model.predict(
            source=image,
            conf=confidence,
            imgsz=IMGSZ,
            device=device,
            verbose=False,
        )[0]
        names = result.names
        detections: list[Detection] = []
        if result.boxes is not None:
            for xyxy, score, class_id in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
            ):
                numeric_class_id = int(class_id)
                detections.append(
                    Detection(
                        class_id=numeric_class_id,
                        class_name=str(names[numeric_class_id]),
                        confidence=round(float(score), 6),
                        box=BoundingBox(
                            x1=round(float(xyxy[0]), 3),
                            y1=round(float(xyxy[1]), 3),
                            x2=round(float(xyxy[2]), 3),
                            y2=round(float(xyxy[3]), 3),
                        ),
                    )
                )
        detections.sort(key=lambda item: item.confidence, reverse=True)
        LOGGER.info(
            "inference_complete width=%d height=%d detections=%d confidence=%.3f imgsz=%d duration_ms=%.1f",
            image.width,
            image.height,
            len(detections),
            confidence,
            IMGSZ,
            (time.perf_counter() - started) * 1000,
        )
        return DetectResponse(
            detections=detections,
            image_size=ImageSize(width=image.width, height=image.height),
            model=_resolved_model_path().name,
            confidence_threshold=confidence,
            imgsz=IMGSZ,
        )
