from __future__ import annotations

import os
import logging
import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

from .model_service import DEFAULT_CONFIDENCE, InvalidImageError, decode_image, detect, get_model
from .reasoning import answer_question, route_question
from .schemas import AskResponse, DetectResponse


app = FastAPI(
    title="RunwayGuard",
    version="1.0.0",
    description="Auditable RT-DETR API for visible airport foreign-object debris.",
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


@app.get("/health")
def health() -> dict[str, str]:
    try:
        get_model()
        return {"status": "ready"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/detect", response_model=DetectResponse)
async def detect_endpoint(
    image: UploadFile = File(...),
    confidence: float = Form(DEFAULT_CONFIDENCE, ge=0.01, le=0.99),
) -> DetectResponse:
    try:
        decoded = decode_image(await read_upload(image))
        return detect(decoded, confidence)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(
    question: str = Form(..., min_length=2, max_length=500),
    image: UploadFile | None = File(None),
    confidence: float = Form(DEFAULT_CONFIDENCE, ge=0.01, le=0.99),
) -> AskResponse:
    route = route_question(question)
    if route != "DETECT":
        return answer_question(question)
    if image is None:
        return AskResponse(
            route="DETECT",
            status="INSUFFICIENT_INFORMATION",
            answer="Insufficient information. This question requires an image, but none was provided.",
            guardrail_reason="Missing image.",
        )
    try:
        decoded = decode_image(await read_upload(image))
        result = detect(decoded, confidence)
        return answer_question(question, result.detections)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
