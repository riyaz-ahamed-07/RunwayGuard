"""Reviewer demo UI helpers: samples, Part B prompts, backend-transparent runs."""

from __future__ import annotations

import base64
import os
import time
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageDraw, ImageFont
from starlette.concurrency import run_in_threadpool

from .model_service import InvalidImageError, decode_image, detect, get_model
from .reasoning import answer_question, route_question
from .runtime_config import (
    ASK_ANSWER_CONFIDENCE,
    ASK_EVIDENCE_CONFIDENCE,
    CLASS_NAMES,
    DETECT_DISPLAY_CONFIDENCE,
    IMGSZ,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = ROOT / "deploy" / "samples"

SAMPLE_CARDS = [
    ("016303.jpg", "Small fastener"),
    ("022564.jpg", "Loose metal"),
    ("022573.jpg", "Metal + debris"),
    ("027929.jpg", "Natural debris"),
    ("027930.jpg", "Natural debris (2)"),
    ("000700.jpg", "Test crop"),
]

PART_B_PROMPTS = [
    {"id": "count_fasteners", "label": "Count fasteners", "text": "How many fasteners are visible?"},
    {"id": "count_hand_tools", "label": "Count hand tools", "text": "How many hand tools are visible?"},
    {"id": "presence_loose_metal", "label": "Presence loose metal", "text": "Is there loose metal in this image?"},
    {"id": "list_detections", "label": "List detections", "text": "What debris candidates are visible?"},
    {"id": "most_common", "label": "Most common class", "text": "What is the most common debris class?"},
    {"id": "highest_conf", "label": "Highest confidence", "text": "Which detection has the highest confidence?"},
    {"id": "subtype_abstain", "label": "Subtype abstain", "text": "Is there a screwdriver?"},
    {"id": "bolt_abstain", "label": "Bolt subtype abstain", "text": "How many bolts are visible?"},
    {"id": "safety_unobs", "label": "Safety unobservable", "text": "Is the runway safe to reopen?"},
    {"id": "clearance_unobs", "label": "Clearance unobservable", "text": "Can aircraft use this runway now?"},
    {"id": "material_unobs", "label": "Material unobservable", "text": "What is the debris made of?"},
    {"id": "off_topic", "label": "Off-topic", "text": "What is the weather forecast tomorrow?"},
]

BOX_COLORS = {
    "fastener_hardware": (46, 134, 222),
    "hand_tool": (231, 76, 60),
    "flexible_debris": (155, 89, 182),
    "loose_metal": (241, 196, 15),
    "plastic_paper_debris": (26, 188, 156),
    "component_container": (230, 126, 34),
    "natural_debris": (39, 174, 96),
}

router = APIRouter(prefix="/demo", tags=["demo"])


async def _read_upload(upload: UploadFile) -> bytes:
    maximum_bytes = int(os.environ.get("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
    content = await upload.read(maximum_bytes + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    if len(content) > maximum_bytes:
        raise HTTPException(status_code=413, detail=f"Image exceeds {maximum_bytes} bytes.")
    return content


def _font(size: int = 14):
    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_detections(image: Image.Image, detections: list, *, min_conf: float) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = _font(14)
    for det in detections:
        if det.confidence < min_conf:
            continue
        color = BOX_COLORS.get(det.class_name, (255, 255, 255))
        box = [det.box.x1, det.box.y1, det.box.x2, det.box.y2]
        draw.rectangle(box, outline=color, width=2)
        label = f"{det.class_name} {det.confidence:.2f}"
        tx, ty = box[0], max(0, box[1] - 16)
        draw.rectangle([tx, ty, tx + 8 * len(label), ty + 16], fill=color)
        draw.text((tx + 2, ty), label, fill=(0, 0, 0), font=font)
    return canvas


def image_to_data_url(image: Image.Image) -> str:
    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=92)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def band_counts(detections: list) -> dict[str, int]:
    return {
        "n_total_returned": len(detections),
        "n_ge_0.25_evidence": sum(1 for d in detections if d.confidence >= ASK_EVIDENCE_CONFIDENCE),
        "n_ge_0.50_answer": sum(1 for d in detections if d.confidence >= ASK_ANSWER_CONFIDENCE),
        "n_between_0.25_and_0.50": sum(
            1 for d in detections if ASK_EVIDENCE_CONFIDENCE <= d.confidence < ASK_ANSWER_CONFIDENCE
        ),
    }


def _model_path() -> str:
    return os.environ.get("MODEL_PATH", "weights/best.pt")


@router.get("/meta")
def demo_meta() -> dict:
    samples = [
        {"file": name, "label": caption, "url": f"/demo/samples/{name}"}
        for name, caption in SAMPLE_CARDS
        if (SAMPLES_DIR / name).exists()
    ]
    model_ok = False
    model_error = None
    try:
        get_model()
        model_ok = True
    except Exception as exc:  # noqa: BLE001 — surface load failure in UI
        model_error = str(exc)

    return {
        "title": "RunwayGuard",
        "subtitle": "Visible FOD candidates · RT-DETR-L · not clearance certification",
        "imgsz": IMGSZ,
        "detect_display_confidence_default": DETECT_DISPLAY_CONFIDENCE,
        "ask_evidence_confidence": ASK_EVIDENCE_CONFIDENCE,
        "ask_answer_confidence": ASK_ANSWER_CONFIDENCE,
        "class_names": list(CLASS_NAMES),
        "model_path": _model_path(),
        "model_ready": model_ok,
        "model_error": model_error,
        "samples": samples,
        "prompts": PART_B_PROMPTS,
        "links": {
            "repo": "https://github.com/riyaz-ahamed-07/RunwayGuard",
            "weights": "https://huggingface.co/DarkKnight1217/RunwayGuard-rtdetr-l",
            "memo": "https://github.com/riyaz-ahamed-07/RunwayGuard/blob/main/MEMO.md",
            "api_docs": "/docs",
            "health": "/health",
        },
    }


@router.get("/samples/{name}")
def get_sample(name: str):
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="Invalid sample name.")
    path = SAMPLES_DIR / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Sample not found.")
    return FileResponse(path, media_type="image/jpeg", filename=name)


def _run_detect(decoded, display_conf: float, pil_image: Image.Image) -> dict:
    started = time.perf_counter()
    wide = detect(decoded, min(0.05, display_conf))
    detect_ms = (time.perf_counter() - started) * 1000
    shown = [d for d in wide.detections if d.confidence >= display_conf]
    annotated = draw_detections(pil_image, wide.detections, min_conf=display_conf)
    return {
        "annotated_image": image_to_data_url(annotated),
        "user": {
            "detections": [d.model_dump() for d in shown],
            "image_size": wide.image_size.model_dump(),
            "confidence_threshold": display_conf,
            "count": len(shown),
        },
        "backend": {
            "endpoint": "/detect",
            "model_path": _model_path(),
            "model_name": wide.model,
            "imgsz": IMGSZ,
            "display_confidence_requested": display_conf,
            "internal_collect_confidence": min(0.05, display_conf),
            "api_detect_default": DETECT_DISPLAY_CONFIDENCE,
            "ask_evidence_confidence": ASK_EVIDENCE_CONFIDENCE,
            "ask_answer_confidence": ASK_ANSWER_CONFIDENCE,
            "class_names": list(CLASS_NAMES),
            "timing_ms": {"detect": round(detect_ms, 1)},
            "score_bands": band_counts(wide.detections),
            "all_detections_unfiltered_for_ui": [d.model_dump() for d in wide.detections],
            "displayed_detections": [d.model_dump() for d in shown],
        },
    }


def _run_ask(question: str, decoded, pil_image: Image.Image | None) -> dict:
    route = route_question(question)
    detect_ms = None
    detection_payload = None
    detections = []
    image_size = None
    annotated = pil_image.convert("RGB") if pil_image is not None else None

    if decoded is not None and route == "DETECT":
        started = time.perf_counter()
        detection_payload = detect(decoded, ASK_EVIDENCE_CONFIDENCE)
        detect_ms = (time.perf_counter() - started) * 1000
        detections = detection_payload.detections
        image_size = detection_payload.image_size
        if pil_image is not None:
            annotated = draw_detections(pil_image, detections, min_conf=ASK_EVIDENCE_CONFIDENCE)

    started = time.perf_counter()
    answer = answer_question(question, detections, image_size=image_size)
    reason_ms = (time.perf_counter() - started) * 1000

    return {
        "annotated_image": None if annotated is None else image_to_data_url(annotated),
        "answer_text": answer.answer,
        "user": {
            "answer": answer.answer,
            "status": answer.status,
            "route": answer.route,
            "intent": answer.intent,
            "guardrail_reason": answer.guardrail_reason,
            "evidence_count": len(answer.evidence),
        },
        "backend": {
            "endpoint": "/ask",
            "model_path": _model_path(),
            "imgsz": IMGSZ,
            "route_precheck": route,
            "ask_evidence_confidence_fixed": ASK_EVIDENCE_CONFIDENCE,
            "ask_answer_confidence": ASK_ANSWER_CONFIDENCE,
            "caller_cannot_lower_evidence_threshold": True,
            "phrasing_backend": answer.phrasing_backend,
            "timing_ms": {
                "detect": None if detect_ms is None else round(detect_ms, 1),
                "reason": round(reason_ms, 1),
            },
            "score_bands_on_evidence_set": band_counts(detections) if detections else None,
            "full_ask_response": answer.model_dump(),
            "detect_payload_used_for_ask": None
            if detection_payload is None
            else {
                "model": detection_payload.model,
                "confidence_threshold": detection_payload.confidence_threshold,
                "imgsz": detection_payload.imgsz,
                "image_size": detection_payload.image_size.model_dump(),
                "detections": [d.model_dump() for d in detection_payload.detections],
            },
        },
    }


@router.post("/detect")
async def demo_detect(
    image: UploadFile = File(...),
    confidence: float = Form(DETECT_DISPLAY_CONFIDENCE, ge=0.01, le=0.99),
) -> dict:
    try:
        raw = await _read_upload(image)
        decoded = decode_image(raw)
        pil = Image.open(BytesIO(raw)).convert("RGB")
        return await run_in_threadpool(_run_detect, decoded, confidence, pil)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/ask")
async def demo_ask(
    question: str = Form(..., min_length=2, max_length=500),
    image: UploadFile | None = File(None),
) -> dict:
    try:
        decoded = None
        pil = None
        if image is not None and image.filename:
            raw = await _read_upload(image)
            decoded = decode_image(raw)
            pil = Image.open(BytesIO(raw)).convert("RGB")
        return await run_in_threadpool(_run_ask, question.strip(), decoded, pil)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
