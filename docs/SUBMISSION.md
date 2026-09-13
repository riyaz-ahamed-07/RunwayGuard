# API and delivery links

Companion to the written memo ([`MEMO.md`](../MEMO.md)).

## Deliverables

| Item | Location |
| --- | --- |
| Source (train, eval, inference, API) | https://github.com/riyaz-ahamed-07/RunwayGuard |
| Live demo (no clone) | https://huggingface.co/spaces/DarkKnight1217/RunwayGuard-demo |
| Weights (`best.pt`) | https://huggingface.co/DarkKnight1217/RunwayGuard-rtdetr-l/resolve/main/best.pt |
| Hub model page | https://huggingface.co/DarkKnight1217/RunwayGuard-rtdetr-l |
| sha256 | `C2D2A418069D9AC8658CA85B8359A9E1AB740CE96826C5138178B080F9243269` |
| Memo | [`MEMO.md`](../MEMO.md) |
| Train log | [`artifacts/results.csv`](artifacts/results.csv) (11 epochs, ~3.9 h, Colab T4) |
| Test metrics | [`artifacts/test_metrics.json`](artifacts/test_metrics.json) (mAP50 0.755, mAP50-95 0.636) |
| Prediction figures | [`figures/predictions/`](figures/predictions/) (input vs `/detect` boxes) |

```bash
hf download DarkKnight1217/RunwayGuard-rtdetr-l best.pt --local-dir weights
```

## Run the API

```bash
git clone https://github.com/riyaz-ahamed-07/RunwayGuard.git
cd RunwayGuard
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
hf download DarkKnight1217/RunwayGuard-rtdetr-l best.pt --local-dir weights
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Reviewer UI (upload, samples, Part B prompts, backend traces): http://localhost:8000/  
OpenAPI: http://localhost:8000/docs

```bash
docker build -t runwayguard .
docker run --rm -p 8000:8000 -e MODEL_PATH=/app/weights/best.pt -e RUNWAYGUARD_LLM=0 runwayguard
```

## Sample requests

Captured from `best.pt` (sha256 above) via `python -m scripts.render_readme_examples`.

### `GET /health`

```bash
curl -s http://localhost:8000/health
```

### `POST /detect` (`deploy/samples/000700.jpg`)

```bash
curl -s -X POST http://localhost:8000/detect \
  -F "image=@deploy/samples/000700.jpg" \
  -F "confidence=0.25"
```

```json
{
  "detections": [
    {
      "class_id": 5,
      "class_name": "component_container",
      "confidence": 0.919957,
      "box": { "x1": 89.59, "y1": 149.152, "x2": 175.272, "y2": 166.513 }
    }
  ],
  "image_size": { "width": 300, "height": 300 },
  "model": "best.pt",
  "confidence_threshold": 0.25,
  "imgsz": 480
}
```

### `POST /ask` (grounded)

```bash
curl -s -X POST http://localhost:8000/ask \
  -F "image=@deploy/samples/000700.jpg" \
  -F "question=How many debris objects are visible?"
```

```json
{
  "route": "DETECT",
  "status": "ANSWERED",
  "answer": "I confidently detected 1 visible debris object(s). This is not proof that the scene is clear of all debris.",
  "evidence": [
    {
      "class_id": 5,
      "class_name": "component_container",
      "confidence": 0.919957,
      "box": { "x1": 89.59, "y1": 149.152, "x2": 175.272, "y2": 166.513 }
    }
  ],
  "guardrail_reason": null,
  "intent": "COUNT_CATEGORY",
  "phrasing_backend": "deterministic"
}
```

### `POST /ask` (unobservable)

```bash
curl -s -X POST http://localhost:8000/ask \
  -F "question=Is the runway safe to reopen?"
```

```json
{
  "route": "UNOBSERVABLE",
  "status": "INSUFFICIENT_INFORMATION",
  "answer": "Insufficient information. One image can support detection of visible candidate debris only; it cannot certify runway safety, predict aircraft damage, or determine hidden physical facts. An authorized inspection is still required.",
  "evidence": [],
  "guardrail_reason": "The question asks for information that is not observable from one image.",
  "intent": "UNOBSERVABLE",
  "phrasing_backend": "deterministic"
}
```

## `/ask` policy

| Rule | Value |
| --- | --- |
| Evidence confidence | fixed at 0.25 |
| Answer confidence | ≥ 0.50 |
| Inference size | imgsz 480 |
| Subtype questions | abstain (e.g. screwdriver vs `hand_tool`) |
| Optional LLM | structured intent JSON only; final text code-rendered; Docker `RUNWAYGUARD_LLM=0` |

## Supporting artifacts

| Artifact | Path |
| --- | --- |
| Live demo UI | https://huggingface.co/spaces/DarkKnight1217/RunwayGuard-demo (free ZeroGPU) |
| Side-by-side predictions | [`figures/predictions/`](figures/predictions/) |
| Failure IDs cited in the memo | [`artifacts/memo_failure_ids.json`](artifacts/memo_failure_ids.json) |
| Failure mining dumps | [`artifacts/failure_mine.json`](artifacts/failure_mine.json), [`artifacts/failure_small_miss.json`](artifacts/failure_small_miss.json) |
| Annotation QC | [`figures/annotation_audit/`](figures/annotation_audit/) |
| Demo UI source | [`../app/static/`](../app/static/), [`../app/demo.py`](../app/demo.py) |
| Gradio (Space) | [`../deploy/gradio_app.py`](../deploy/gradio_app.py) |
| Colab notebook | [`../notebooks/colab_train.ipynb`](../notebooks/colab_train.ipynb) |
| Hub model card | [`hf/MODEL_CARD.md`](hf/MODEL_CARD.md) |
