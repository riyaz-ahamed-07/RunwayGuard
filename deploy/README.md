# Gradio / HF Spaces demo for RunwayGuard

## Option A — local

```bash
pip install gradio huggingface_hub
set HF_WEIGHTS_REPO=DarkKnight1217/RunwayGuard-rtdetr-l
# or place weights/best.pt locally and skip the env var
set PYTHONPATH=.
python deploy/gradio_app.py
```

## Option B — Hugging Face Space

1. Create Space → Gradio → Python 3.11 (public).
2. Add Space variable: `HF_WEIGHTS_REPO=DarkKnight1217/RunwayGuard-rtdetr-l`.
3. Upload / sync: `app/`, `config/`, `deploy/gradio_app.py` as `app.py` entry, plus a Space `requirements.txt`:

```text
ultralytics==8.4.147
pillow==11.3.0
pydantic==2.11.7
httpx==0.28.1
gradio>=4.44.0
huggingface_hub>=0.25.0
python-multipart==0.0.20
fastapi==0.116.1
```

4. Point Space to run `deploy/gradio_app.py` (or copy it to root `app.py` and fix imports).

## Option C — FastAPI for reviewers

```bash
docker build -t runwayguard .
docker run --rm -p 8000:8000 -e MODEL_PATH=/app/weights/best.pt runwayguard
```

Share `https://YOUR_HOST/docs` if the API is hosted publicly.
