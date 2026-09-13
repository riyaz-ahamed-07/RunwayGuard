"""RunwayGuard reviewer UI (Gradio).

Loads weights from local MODEL_PATH or HF_WEIGHTS_REPO (default DarkKnight1217/RunwayGuard-rtdetr-l).
Shows user-facing answers plus backend thresholds, timings, evidence bands, and raw JSON.
"""

from __future__ import annotations

import json
import os
import time
from io import BytesIO
from pathlib import Path

import gradio as gr
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont

from app.model_service import detect, decode_image, get_model
from app.reasoning import answer_question, route_question
from app.runtime_config import (
    ASK_ANSWER_CONFIDENCE,
    ASK_EVIDENCE_CONFIDENCE,
    CLASS_NAMES,
    DETECT_DISPLAY_CONFIDENCE,
    IMGSZ,
)

DEFAULT_WEIGHTS_REPO = "DarkKnight1217/RunwayGuard-rtdetr-l"
SAMPLES_DIR = Path(__file__).resolve().parent / "samples"

SAMPLE_CARDS = [
    ("016303.jpg", "Tiny fastener"),
    ("022564.jpg", "Loose metal"),
    ("022573.jpg", "Metal + debris"),
    ("027929.jpg", "Organic debris"),
    ("027930.jpg", "Organic vs metal"),
    ("000700.jpg", "Battery crop"),
]

PART_B_PROMPTS = [
    ("Count fasteners", "How many fasteners are visible?"),
    ("Count hand tools", "How many hand tools are visible?"),
    ("Presence loose metal", "Is there loose metal in this image?"),
    ("List detections", "What debris candidates are visible?"),
    ("Most common class", "What is the most common debris class?"),
    ("Highest confidence", "Which detection has the highest confidence?"),
    ("Subtype abstain", "Is there a screwdriver?"),
    ("Bolt subtype abstain", "How many bolts are visible?"),
    ("Safety unobservable", "Is the runway safe to reopen?"),
    ("Clearance unobservable", "Can aircraft use this runway now?"),
    ("Material unobservable", "What is the debris made of?"),
    ("Off-topic", "What is the weather forecast tomorrow?"),
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


def ensure_weights() -> str:
    target = Path(os.environ.get("MODEL_PATH", "weights/best.pt"))
    if target.exists():
        os.environ["MODEL_PATH"] = str(target)
        return str(target)
    repo = os.environ.get("HF_WEIGHTS_REPO", DEFAULT_WEIGHTS_REPO).strip()
    downloaded = hf_hub_download(repo_id=repo, filename="best.pt")
    os.environ["MODEL_PATH"] = downloaded
    return downloaded


WEIGHTS_PATH = ensure_weights()
get_model()


def _font(size: int = 14):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
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


def image_to_bytes(image: Image.Image) -> bytes:
    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def band_counts(detections: list) -> dict[str, int]:
    return {
        "n_total_returned": len(detections),
        "n_ge_0.25_evidence": sum(1 for d in detections if d.confidence >= ASK_EVIDENCE_CONFIDENCE),
        "n_ge_0.50_answer": sum(1 for d in detections if d.confidence >= ASK_ANSWER_CONFIDENCE),
        "n_between_0.25_and_0.50": sum(
            1 for d in detections if ASK_EVIDENCE_CONFIDENCE <= d.confidence < ASK_ANSWER_CONFIDENCE
        ),
    }


def load_sample(choice: str):
    if not choice:
        return None
    name = choice.split(" ", 1)[0]
    path = SAMPLES_DIR / name
    if not path.exists():
        return None
    return Image.open(path).convert("RGB")


def apply_prompt(label: str) -> str:
    for name, text in PART_B_PROMPTS:
        if name == label:
            return text
    return ""


def run_detect_ui(image: Image.Image | None, display_conf: float):
    if image is None:
        empty = {"error": "Upload an image or pick a sample."}
        return None, json.dumps(empty, indent=2), json.dumps(empty, indent=2)

    started = time.perf_counter()
    decoded = decode_image(image_to_bytes(image))
    # Pull a wide set for transparency, then filter display.
    wide = detect(decoded, min(0.05, display_conf))
    detect_ms = (time.perf_counter() - started) * 1000
    shown = [d for d in wide.detections if d.confidence >= display_conf]
    annotated = draw_detections(image, wide.detections, min_conf=display_conf)

    user_view = {
        "detections": [d.model_dump() for d in shown],
        "image_size": wide.image_size.model_dump(),
        "confidence_threshold": display_conf,
        "count": len(shown),
    }
    backend = {
        "endpoint": "/detect",
        "model_path": WEIGHTS_PATH,
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
    }
    return annotated, json.dumps(user_view, indent=2), json.dumps(backend, indent=2)


def run_ask_ui(image: Image.Image | None, question: str):
    question = (question or "").strip()
    if not question:
        empty = {"error": "Enter or select a Part B prompt."}
        return None, "", json.dumps(empty, indent=2), json.dumps(empty, indent=2)

    route = route_question(question)
    detect_ms = None
    detection_payload = None
    detections = []
    image_size = None
    annotated = None

    if image is not None and route == "DETECT":
        started = time.perf_counter()
        decoded = decode_image(image_to_bytes(image))
        # Match API: /ask evidence is fixed at 0.25
        detection_payload = detect(decoded, ASK_EVIDENCE_CONFIDENCE)
        detect_ms = (time.perf_counter() - started) * 1000
        detections = detection_payload.detections
        image_size = detection_payload.image_size
        annotated = draw_detections(image, detections, min_conf=ASK_EVIDENCE_CONFIDENCE)
    elif image is not None:
        annotated = image.convert("RGB")

    started = time.perf_counter()
    answer = answer_question(question, detections, image_size=image_size)
    reason_ms = (time.perf_counter() - started) * 1000

    user_view = {
        "answer": answer.answer,
        "status": answer.status,
        "route": answer.route,
        "intent": answer.intent,
        "guardrail_reason": answer.guardrail_reason,
        "evidence_count": len(answer.evidence),
    }
    backend = {
        "endpoint": "/ask",
        "model_path": WEIGHTS_PATH,
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
    }
    return (
        annotated,
        answer.answer,
        json.dumps(user_view, indent=2),
        json.dumps(backend, indent=2),
    )


SAMPLE_CHOICES = [f"{name} - {caption}" for name, caption in SAMPLE_CARDS if (SAMPLES_DIR / name).exists()]
PROMPT_CHOICES = [name for name, _ in PART_B_PROMPTS]


with gr.Blocks(title="RunwayGuard") as demo:
    gr.Markdown(
        f"""
# RunwayGuard
Visible airport FOD candidates (RT-DETR-L). **Not** runway clearance certification.

| Setting | Value |
| --- | --- |
| imgsz | `{IMGSZ}` |
| `/detect` default display conf | `{DETECT_DISPLAY_CONFIDENCE}` |
| `/ask` evidence conf (fixed) | `{ASK_EVIDENCE_CONFIDENCE}` |
| `/ask` answer conf | `{ASK_ANSWER_CONFIDENCE}` |
| weights | `{Path(WEIGHTS_PATH).name}` |
| classes | {", ".join(CLASS_NAMES)} |

Repo: [riyaz-ahamed-07/RunwayGuard](https://github.com/riyaz-ahamed-07/RunwayGuard) ·
Weights: [DarkKnight1217/RunwayGuard-rtdetr-l](https://huggingface.co/DarkKnight1217/RunwayGuard-rtdetr-l) ·
Memo: [MEMO.md](https://github.com/riyaz-ahamed-07/RunwayGuard/blob/main/MEMO.md)
"""
    )

    with gr.Row():
        sample_dd = gr.Dropdown(choices=SAMPLE_CHOICES, label="Load sample image (memo failure IDs + test crop)")
        load_btn = gr.Button("Load sample", variant="secondary")

    with gr.Tab("Detect"):
        with gr.Row():
            with gr.Column():
                img_d = gr.Image(type="pil", label="Upload or sample", height=360)
                conf_d = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Display confidence (/detect filter)")
                run_d = gr.Button("Run /detect", variant="primary")
            with gr.Column():
                out_img_d = gr.Image(type="pil", label="Boxes (display threshold)", height=360)
        with gr.Row():
            user_d = gr.Code(language="json", label="User-facing detections")
            backend_d = gr.Code(language="json", label="Backend trace (thresholds, bands, timings, full set)")
        run_d.click(run_detect_ui, [img_d, conf_d], [out_img_d, user_d, backend_d])

    with gr.Tab("Ask (Part B)"):
        with gr.Row():
            with gr.Column():
                img_a = gr.Image(type="pil", label="Upload or sample (optional for unobservable prompts)", height=320)
                prompt_dd = gr.Dropdown(choices=PROMPT_CHOICES, label="Ready Part B prompts", value="Count fasteners")
                fill_btn = gr.Button("Paste prompt into box")
                question = gr.Textbox(
                    label="Question",
                    value="How many fasteners are visible?",
                    lines=2,
                )
                run_a = gr.Button("Run /ask", variant="primary")
            with gr.Column():
                out_img_a = gr.Image(type="pil", label="Evidence boxes (≥ 0.25)", height=320)
                answer_box = gr.Textbox(label="Answer text", lines=4)
        with gr.Row():
            user_a = gr.Code(language="json", label="User-facing ask summary")
            backend_a = gr.Code(language="json", label="Backend trace (route, fixed thresholds, evidence, timings)")
        fill_btn.click(apply_prompt, [prompt_dd], [question])
        prompt_dd.change(apply_prompt, [prompt_dd], [question])
        run_a.click(run_ask_ui, [img_a, question], [out_img_a, answer_box, user_a, backend_a])

    load_btn.click(load_sample, [sample_dd], [img_d])
    load_btn.click(load_sample, [sample_dd], [img_a])
    sample_dd.change(load_sample, [sample_dd], [img_d])
    sample_dd.change(load_sample, [sample_dd], [img_a])


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
    )
