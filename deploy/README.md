# RunwayGuard reviewer demo

## What you get

Open **`/`** for the UI:

- Upload an image **or** load memo failure samples (`016303`, `022564`, …)
- **Detect** tab: boxes + user JSON + backend trace (thresholds, score bands, timings, full detection set)
- **Ask (Part B)** tab: ready-to-paste prompts + answer + backend route/evidence/timing JSON
- Official API still at `/docs`, `/detect`, `/ask`, `/health`

## Local

```bash
pip install -r requirements.txt
# weights/best.pt already present, or:
# set HF_WEIGHTS_REPO=DarkKnight1217/RunwayGuard-rtdetr-l
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/

## Docker

```bash
docker build -t runwayguard .
docker run --rm -p 8000:8000 -e RUNWAYGUARD_LLM=0 runwayguard
```

If `weights/` is empty in the image, set `HF_WEIGHTS_REPO=DarkKnight1217/RunwayGuard-rtdetr-l` so startup downloads `best.pt`.

## Public URL (temporary tunnel)

While the API is running locally:

```bash
npx --yes localtunnel --port 8000
```

Share the printed URL. Keep the machine awake while reviewers test.

## Hugging Face Gradio Space

HF now requires **PRO** for free `cpu-basic` Gradio/Docker Spaces. Prefer the FastAPI UI above, or pay for PRO and use `deploy/hf_space/` + `scripts/deploy_hf_space.ps1`.

## Gradio (local only)

```bash
pip install "gradio>=5.5.0" huggingface_hub
set PYTHONPATH=.
python deploy/gradio_app.py
```
