"""Render genuine model predictions for README side-by-side examples.

Run from repo root:
  PYTHONPATH=. python -m scripts.render_readme_examples

Draws boxes from /detect only (not ground-truth annotations).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ("000700", "022564", "016303")
COLORS = (
    (46, 134, 222),
    (231, 76, 60),
    (241, 196, 15),
    (39, 174, 96),
    (155, 89, 182),
)


def _font(size: int = 14):
    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_boxes(image: Image.Image, detections: list) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = _font(13)
    for index, det in enumerate(detections):
        color = COLORS[index % len(COLORS)]
        box = det["box"]
        xy = [box["x1"], box["y1"], box["x2"], box["y2"]]
        draw.rectangle(xy, outline=color, width=2)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        tx, ty = xy[0], max(0, xy[1] - 16)
        draw.rectangle([tx, ty, tx + 7 * len(label), ty + 16], fill=color)
        draw.text((tx + 2, ty), label, fill=(0, 0, 0), font=font)
    return canvas


def panel(title: str, image: Image.Image, *, note: str = "") -> Image.Image:
    pad = 10
    header = 28
    footer = 22 if note else 0
    frame = Image.new("RGB", (image.width + pad * 2, image.height + header + footer + pad), "#f4f4f1")
    draw = ImageDraw.Draw(frame)
    font = _font(14)
    small = _font(11)
    draw.rectangle([0, 0, frame.width, header], fill="#1a1a1a")
    draw.text((pad, 6), title, fill="#ffffff", font=font)
    frame.paste(image, (pad, header))
    if note:
        draw.text((pad, header + image.height + 4), note, fill="#444444", font=small)
    return frame


def side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    gap = 12
    canvas = Image.new("RGB", (left.width + right.width + gap, max(left.height, right.height)), "#e8e8e4")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))
    return canvas


def main() -> None:
    os.environ["RUNWAYGUARD_LLM"] = "0"
    from fastapi.testclient import TestClient

    from app.main import app

    out = ROOT / "docs" / "figures" / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(os.environ.get("MODEL_PATH", ROOT / "weights" / "best.pt"))
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest().upper()
    client = TestClient(app)
    records: dict = {
        "weights_sha256": digest,
        "imgsz": 480,
        "confidence": 0.25,
        "examples": [],
    }

    for stem in SAMPLES:
        source = ROOT / "deploy" / "samples" / f"{stem}.jpg"
        content = source.read_bytes()
        response = client.post(
            "/detect",
            data={"confidence": "0.25"},
            files={"image": (source.name, content, "image/jpeg")},
        )
        response.raise_for_status()
        payload = response.json()
        original = Image.open(source).convert("RGB")
        predicted = draw_boxes(original, payload["detections"])
        n = len(payload["detections"])
        left = panel(f"{stem} · input", original)
        right = panel(
            f"{stem} · model @ conf≥0.25",
            predicted,
            note=f"{n} box(es) from /detect · not ground truth",
        )
        combo = side_by_side(left, right)
        combo_path = out / f"{stem}_side_by_side.png"
        combo.save(combo_path)
        predicted.save(out / f"{stem}_predicted.png")
        original.save(out / f"{stem}_input.png")

        ask = client.post(
            "/ask",
            data={"question": "How many debris objects are visible?"},
            files={"image": (source.name, content, "image/jpeg")},
        )
        ask.raise_for_status()
        records["examples"].append(
            {
                "image_id": stem,
                "source_sha256": hashlib.sha256(content).hexdigest().upper(),
                "figure": str(combo_path.relative_to(ROOT)).replace("\\", "/"),
                "detect": payload,
                "ask": ask.json(),
            }
        )
        print(stem, "detections=", n, flush=True)

    safety = client.post("/ask", data={"question": "Is the runway safe to reopen?"})
    safety.raise_for_status()
    records["safety_example"] = safety.json()
    (out / "responses.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
