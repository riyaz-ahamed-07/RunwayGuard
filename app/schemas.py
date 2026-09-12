from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    box: BoundingBox


class ImageSize(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class DetectResponse(BaseModel):
    detections: list[Detection]
    image_size: ImageSize
    model: str
    confidence_threshold: float


class AskResponse(BaseModel):
    route: Literal["DETECT", "NO_DETECTION_NEEDED", "UNOBSERVABLE"]
    status: Literal["ANSWERED", "INSUFFICIENT_INFORMATION"]
    answer: str
    evidence: list[Detection] = Field(default_factory=list)
    guardrail_reason: str | None = None
