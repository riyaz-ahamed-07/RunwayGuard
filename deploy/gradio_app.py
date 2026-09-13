"""Lightweight Gradio demo for RAP reviewers (Hugging Face Spaces).

Set Space secrets / env:
  HF_WEIGHTS_REPO = your-user/RunwayGuard-rtdetr-l
  MODEL_PATH is set automatically after download.
"""

from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path

import gradio as gr
from huggingface_hub import hf_hub_download
from PIL import Image

from app.model_service import detect, decode_image
from app.reasoning import answer_question


def ensure_weights() -> Path:
    target = Path(os.environ.get("MODEL_PATH", "weights/best.pt"))
    if target.exists():
        return target
    repo = os.environ.get("HF_WEIGHTS_REPO", "").strip()
    if not repo:
        raise FileNotFoundError(
            "weights/best.pt missing and HF_WEIGHTS_REPO is unset. "
            "Upload best.pt to a Hub model repo and set HF_WEIGHTS_REPO."
        )
    downloaded = hf_hub_download(repo_id=repo, filename="best.pt")
    os.environ["MODEL_PATH"] = downloaded
    return Path(downloaded)


ensure_weights()


def run_detect(image: Image.Image, confidence: float) -> str:
    if image is None:
        return "Upload an image."
    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG")
    decoded = decode_image(buf.getvalue())
    result = detect(decoded, confidence)
    return json.dumps(result.model_dump(), indent=2)


def run_ask(image: Image.Image | None, question: str, confidence: float) -> str:
    question = (question or "").strip()
    if not question:
        return "Enter a question."
    if image is None:
        answer = answer_question(question, [], image_size=None)
        return json.dumps(answer.model_dump(), indent=2)
    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG")
    decoded = decode_image(buf.getvalue())
    detection = detect(decoded, confidence)
    answer = answer_question(question, detection.detections, image_size=detection.image_size)
    return json.dumps(answer.model_dump(), indent=2)


with gr.Blocks(title="RunwayGuard") as demo:
    gr.Markdown(
        "# RunwayGuard\n"
        "Visible FOD candidates via RT-DETR. "
        "Does **not** certify runway clearance."
    )
    with gr.Tab("Detect"):
        img_d = gr.Image(type="pil", label="Image")
        conf_d = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Display confidence")
        out_d = gr.Code(language="json", label="Response")
        gr.Button("Run /detect").click(run_detect, [img_d, conf_d], out_d)
    with gr.Tab("Ask"):
        img_a = gr.Image(type="pil", label="Image (optional for unobservable questions)")
        q = gr.Textbox(label="Question", value="How many fasteners are visible?")
        conf_a = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Evidence confidence (API fixes ask evidence at 0.25)")
        out_a = gr.Code(language="json", label="Response")
        gr.Button("Run /ask").click(run_ask, [img_a, q, conf_a], out_a)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
