# RunwayGuard reviewer demo

## Public link (share this)

https://huggingface.co/spaces/DarkKnight1217/RunwayGuard-demo

Free ZeroGPU Gradio Space — reviewers open it in a browser (no clone). First wake can be slow; free visitors have a small daily GPU quota.

Redeploy:

```bash
python scripts/deploy_hf_space.py
```

## Local FastAPI UI

```bash
pip install -r requirements.txt
# weights/best.pt already present, or:
# set HF_WEIGHTS_REPO=DarkKnight1217/RunwayGuard-rtdetr-l
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/

- Upload or pick `Image ######` samples
- Detect / Ask with Part B prompts
- User JSON + backend traces (thresholds, bands, timings)

## Docker

```bash
docker build -t runwayguard .
docker run --rm -p 8000:8000 -e RUNWAYGUARD_LLM=0 runwayguard
```

If `weights/` is empty in the image, set `HF_WEIGHTS_REPO=DarkKnight1217/RunwayGuard-rtdetr-l`.

## Gradio (local only)

```bash
pip install "gradio>=5.5.0" huggingface_hub spaces
set PYTHONPATH=.
python deploy/gradio_app.py
```
