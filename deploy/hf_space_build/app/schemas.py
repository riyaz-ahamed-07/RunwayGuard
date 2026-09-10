from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def ensure_ordered_finite_box(self) -> BoundingBox:
        for value in (self.x1, self.y1, self.x2, self.y2):
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError("Bounding box coordinates must be finite.")
        x1, x2 = sorted((self.x1, self.x2))
        y1, y2 = sorted((self.y1, self.y2))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Bounding box must have positive area.")
        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2
        return self


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
    imgsz: int


class AskResponse(BaseModel):
    route: Literal["DETECT", "NO_DETECTION_NEEDED", "UNOBSERVABLE"]
    status: Literal["ANSWERED", "INSUFFICIENT_INFORMATION"]
    answer: str
    evidence: list[Detection] = Field(default_factory=list)
    guardrail_reason: str | None = None
    intent: str | None = None
    phrasing_backend: str = "deterministic"
