from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from .model_service import DEFAULT_CONFIDENCE, InvalidImageError, decode_image, detect, get_model
from .reasoning import CONFIDENT_THRESHOLD, answer_question, route_question
from .runtime_config import ASK_EVIDENCE_CONFIDENCE, DETECT_DISPLAY_CONFIDENCE, IMGSZ
from .schemas import AskResponse, DetectResponse


app = FastAPI(
    title="RunwayGuard",
    version="1.2.0",
    description=(
        "Auditable RT-DETR API for visible airport foreign-object debris. "
        f"/detect display confidence default={DETECT_DISPLAY_CONFIDENCE}; "
        f"/ask evidence threshold={ASK_EVIDENCE_CONFIDENCE} (fixed); "
        f"Part B answer threshold={CONFIDENT_THRESHOLD}; imgsz={IMGSZ}."
    ),
)
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("runwayguard.api")


@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        LOGGER.exception("request_failed id=%s method=%s path=%s", request_id, request.method, request.url.path)
        raise
    response.headers["x-request-id"] = request_id
    LOGGER.info(
        "request_complete id=%s method=%s path=%s status=%d duration_ms=%.1f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - started) * 1000,
    )
    return response


async def read_upload(upload: UploadFile) -> bytes:
    maximum_bytes = int(os.environ.get("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
    content = await upload.read(maximum_bytes + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    if len(content) > maximum_bytes:
        raise HTTPException(status_code=413, detail=f"Image exceeds {maximum_bytes} bytes.")
    return content


def _ask_with_detections(question: str, detections, image_size) -> AskResponse:
    return answer_question(question, detections, image_size=image_size)


@app.get("/health")
def health() -> dict[str, object]:
    try:
        get_model()
        return {
            "status": "ready",
            "model": Path(os.environ.get("MODEL_PATH", "weights/best.pt")).name,
            "imgsz": IMGSZ,
            "detect_display_confidence_default": DETECT_DISPLAY_CONFIDENCE,
            "ask_evidence_confidence": ASK_EVIDENCE_CONFIDENCE,
            "ask_answer_confidence": CONFIDENT_THRESHOLD,
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/detect", response_model=DetectResponse)
async def detect_endpoint(
    image: UploadFile = File(...),
    confidence: float = Form(DEFAULT_CONFIDENCE, ge=0.01, le=0.99),
) -> DetectResponse:
    try:
        decoded = decode_image(await read_upload(image))
        return await run_in_threadpool(detect, decoded, confidence)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(
    question: str = Form(..., min_length=2, max_length=500),
    image: UploadFile | None = File(None),
    confidence: float | None = Form(None, ge=0.01, le=0.99),
) -> AskResponse:
    del confidence
    route = route_question(question)
    if route != "DETECT":
        return await run_in_threadpool(answer_question, question)
    if image is None:
        return AskResponse(
            route="DETECT",
            status="INSUFFICIENT_INFORMATION",
            answer="Insufficient information. This question requires an image, but none was provided.",
            guardrail_reason="Missing image.",
            intent="MISSING_IMAGE",
            phrasing_backend="deterministic",
        )
    try:
        decoded = decode_image(await read_upload(image))
        result = await run_in_threadpool(detect, decoded, ASK_EVIDENCE_CONFIDENCE)
        return await run_in_threadpool(_ask_with_detections, question, result.detections, result.image_size)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
