"""RunwayGuard reviewer UI (Gradio) — single-viewport, Space-friendly.

Loads weights from local MODEL_PATH or HF_WEIGHTS_REPO.

On Hugging Face Spaces, GRADIO_SSR_MODE must be false (env var). SSR bakes
image URLs as http://0.0.0.0:7860 and the browser blocks them.
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

try:
    import spaces
except ImportError:  # local / non-ZeroGPU

    class _SpacesShim:
        @staticmethod
        def GPU(fn=None, **_kwargs):
            if fn is None:
                return lambda f: f
            return fn

    spaces = _SpacesShim()

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
    ("000700.jpg", "Image 000700"),
    ("022564.jpg", "Image 022564"),
    ("027929.jpg", "Image 027929"),
    ("027930.jpg", "Image 027930"),
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
_MODEL = get_model()
try:
    import torch

    if torch.cuda.is_available() and hasattr(_MODEL, "model"):
        _MODEL.model.to("cuda")
except Exception:  # noqa: BLE001
    pass


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


def image_to_bytes(image: Image.Image) -> bytes:
    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def as_pil(image) -> Image.Image | None:
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def band_counts(detections: list) -> dict[str, int]:
    return {
        "n_total_returned": len(detections),
        "n_ge_0.25_evidence": sum(1 for d in detections if d.confidence >= ASK_EVIDENCE_CONFIDENCE),
        "n_ge_0.50_answer": sum(1 for d in detections if d.confidence >= ASK_ANSWER_CONFIDENCE),
        "n_between_0.25_and_0.50": sum(
            1 for d in detections if ASK_EVIDENCE_CONFIDENCE <= d.confidence < ASK_ANSWER_CONFIDENCE
        ),
    }


SAMPLE_CHOICES = [
    (caption, name) for name, caption in SAMPLE_CARDS if (SAMPLES_DIR / name).is_file()
]
DEFAULT_SAMPLE = SAMPLE_CHOICES[0][1] if SAMPLE_CHOICES else None
PROMPT_CHOICES = [name for name, _ in PART_B_PROMPTS]
PROMPT_MAP = dict(PART_B_PROMPTS)


def sample_path(filename: str | None) -> str | None:
    if not filename:
        return None
    path = SAMPLES_DIR / filename
    return str(path.resolve()) if path.is_file() else None


def open_sample(filename: str | None) -> Image.Image | None:
    path = sample_path(filename)
    if not path:
        return None
    return Image.open(path).convert("RGB")


def load_sample(filename: str | None):
    path = sample_path(filename)
    image = open_sample(filename)
    return path, image, f"Loaded {filename}" if path else "Sample not found."


def on_upload(image):
    pil = as_pil(image)
    if pil is None:
        return None, None, "No image."
    return pil, pil, "Upload ready."


def apply_prompt(label: str) -> str:
    return PROMPT_MAP.get(label, "")


@spaces.GPU(duration=120)
def run_detect_ui(image, display_conf: float):
    pil = as_pil(image)
    if pil is None:
        empty = {"error": "Choose a sample or upload an image first."}
        return None, json.dumps(empty, indent=2), json.dumps(empty, indent=2), "Choose an image first."

    started = time.perf_counter()
    decoded = decode_image(image_to_bytes(pil))
    wide = detect(decoded, min(0.05, display_conf))
    detect_ms = (time.perf_counter() - started) * 1000
    shown = [d for d in wide.detections if d.confidence >= display_conf]
    annotated = draw_detections(pil, wide.detections, min_conf=display_conf)

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
    status = f"Detect · {detect_ms:.0f} ms · {len(shown)} boxes ≥ {display_conf:.2f}"
    return annotated, json.dumps(user_view, indent=2), json.dumps(backend, indent=2), status


@spaces.GPU(duration=120)
def run_ask_ui(image, question: str):
    question = (question or "").strip()
    if not question:
        empty = {"error": "Enter or select a Part B prompt."}
        return None, "", json.dumps(empty, indent=2), json.dumps(empty, indent=2), "Enter a question."

    pil = as_pil(image)
    route = route_question(question)
    detect_ms = None
    detection_payload = None
    detections = []
    image_size = None
    annotated = pil.copy() if pil is not None else None

    if pil is not None and route == "DETECT":
        started = time.perf_counter()
        decoded = decode_image(image_to_bytes(pil))
        detection_payload = detect(decoded, ASK_EVIDENCE_CONFIDENCE)
        detect_ms = (time.perf_counter() - started) * 1000
        detections = detection_payload.detections
        image_size = detection_payload.image_size
        annotated = draw_detections(pil, detections, min_conf=ASK_EVIDENCE_CONFIDENCE)

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
    status = f"Ask · {route} · detect {detect_ms if detect_ms is not None else '—'} ms · reason {reason_ms:.0f} ms"
    return (
        annotated,
        answer.answer,
        json.dumps(user_view, indent=2),
        json.dumps(backend, indent=2),
        status,
    )


CSS = """
.gradio-container { max-width: 920px !important; margin: 0 auto !important; }
#stage-image { min-height: 320px; }
#stage-image img {
  object-fit: contain !important;
  width: 100% !important;
  max-height: 48vh !important;
  background: #111;
}
#status-line { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.85rem; opacity: 0.85; }
.compact-row { gap: 0.75rem !important; align-items: end !important; }
"""

# Spaces SSR can bake img src as http://0.0.0.0:7860/... — rewrite to same-origin paths.
HEAD_JS = """
<script>
(function () {
  function fixUrls(root) {
    (root || document).querySelectorAll("img, a, source").forEach(function (el) {
      ["src", "href"].forEach(function (attr) {
        var v = el.getAttribute && el.getAttribute(attr);
        if (!v) return;
        if (v.indexOf("0.0.0.0") === -1 && v.indexOf("127.0.0.1") === -1) return;
        try {
          var u = new URL(v, window.location.origin);
          el.setAttribute(attr, u.pathname + u.search);
        } catch (e) {}
      });
    });
  }
  fixUrls(document);
  new MutationObserver(function (mutations) {
    mutations.forEach(function (m) {
      m.addedNodes.forEach(function (n) {
        if (n.nodeType === 1) fixUrls(n);
      });
    });
  }).observe(document.documentElement, { childList: true, subtree: true });
})();
</script>
"""

DEFAULT_PATH = sample_path(DEFAULT_SAMPLE)
DEFAULT_PIL = open_sample(DEFAULT_SAMPLE)

with gr.Blocks(title="RunwayGuard", css=CSS, head=HEAD_JS, theme=gr.themes.Soft()) as demo:
    original = gr.State(DEFAULT_PIL)

    gr.Markdown(
        f"""
# RunwayGuard
Visible FOD candidates · RT-DETR-L · **not** clearance certification  
`imgsz {IMGSZ}` · detect `{DETECT_DISPLAY_CONFIDENCE}` · evidence `{ASK_EVIDENCE_CONFIDENCE}` · answer `{ASK_ANSWER_CONFIDENCE}` · `{Path(WEIGHTS_PATH).name}`  
[Repo](https://github.com/riyaz-ahamed-07/RunwayGuard) · [Weights](https://huggingface.co/DarkKnight1217/RunwayGuard-rtdetr-l) · [Memo](https://github.com/riyaz-ahamed-07/RunwayGuard/blob/main/MEMO.md)
"""
    )

    with gr.Row(elem_classes=["compact-row"]):
        sample_dd = gr.Dropdown(
            choices=SAMPLE_CHOICES,
            value=DEFAULT_SAMPLE,
            label="1 · Choose sample",
            scale=3,
        )
        upload = gr.Image(
            type="pil",
            label="Or upload",
            height=88,
            sources=["upload"],
            scale=2,
        )

    stage = gr.Image(
        value=DEFAULT_PATH,
        type="pil",
        label="2 · Image / detections",
        elem_id="stage-image",
        height=380,
        show_download_button=True,
    )
    status = gr.Markdown("Pick a sample → Run detect or Ask.", elem_id="status-line")

    with gr.Tab("Detect"):
        conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Display confidence")
        run_d = gr.Button("Run detect", variant="primary")
    with gr.Tab("Ask"):
        prompt_dd = gr.Dropdown(choices=PROMPT_CHOICES, value="Count fasteners", label="Part B prompt")
        question = gr.Textbox(value="How many fasteners are visible?", label="Question", lines=2)
        run_a = gr.Button("Run ask", variant="primary")
        answer_box = gr.Textbox(label="Answer", lines=3)

    with gr.Accordion("JSON responses", open=False):
        user_json = gr.Code(language="json", label="User response", max_lines=16)
        backend_json = gr.Code(language="json", label="Backend trace", max_lines=16)

    sample_dd.change(load_sample, [sample_dd], [stage, original, status])
    upload.change(on_upload, [upload], [stage, original, status])
    prompt_dd.change(apply_prompt, [prompt_dd], [question])
    demo.load(load_sample, [sample_dd], [stage, original, status])

    run_d.click(run_detect_ui, [original, conf], [stage, user_json, backend_json, status])
    run_a.click(run_ask_ui, [original, question], [stage, answer_box, user_json, backend_json, status])


def _launch(**kwargs):
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        allowed_paths=[str(SAMPLES_DIR.resolve())],
        **kwargs,
    )


if __name__ == "__main__":
    _launch()
