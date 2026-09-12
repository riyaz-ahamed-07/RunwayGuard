from __future__ import annotations

import io
import logging
import os
import time
from functools import lru_cache
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from ultralytics import RTDETR

from .schemas import BoundingBox, DetectResponse, Detection, ImageSize


DEFAULT_CONFIDENCE = 0.25
LOGGER = logging.getLogger("runwayguard.inference")


class InvalidImageError(ValueError):
    pass


def decode_image(content: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(content))
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("The uploaded file is not a readable image.") from exc


@lru_cache(maxsize=1)
def get_model() -> RTDETR:
    model_path = Path(os.environ.get("MODEL_PATH", "weights/best.pt"))
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. Set MODEL_PATH to the trained best.pt file."
        )
    return RTDETR(str(model_path))


def detect(image: Image.Image, confidence: float = DEFAULT_CONFIDENCE) -> DetectResponse:
    model = get_model()
    started = time.perf_counter()
    result = model.predict(source=image, conf=confidence, verbose=False)[0]
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
        "inference_complete width=%d height=%d detections=%d confidence=%.3f duration_ms=%.1f",
        image.width,
        image.height,
        len(detections),
        confidence,
        (time.perf_counter() - started) * 1000,
    )
    return DetectResponse(
        detections=detections,
        image_size=ImageSize(width=image.width, height=image.height),
        model=Path(os.environ.get("MODEL_PATH", "weights/best.pt")).name,
        confidence_threshold=confidence,
    )
