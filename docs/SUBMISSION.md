# API and delivery links

Companion to the written memo ([`MEMO.md`](../MEMO.md)).

## Deliverables

| Item | Location |
| --- | --- |
| Source (train, eval, inference, API) | https://github.com/riyaz-ahamed-07/RunwayGuard |
| Weights (`best.pt`) | https://huggingface.co/DarkKnight1217/RunwayGuard-rtdetr-l/resolve/main/best.pt |
| Hub model page | https://huggingface.co/DarkKnight1217/RunwayGuard-rtdetr-l |
| sha256 | `C2D2A418069D9AC8658CA85B8359A9E1AB740CE96826C5138178B080F9243269` |
| Memo | [`MEMO.md`](../MEMO.md) |
| Train log | [`artifacts/results.csv`](artifacts/results.csv) (11 epochs, ~3.9 h, Colab T4) |
| Test metrics | [`artifacts/test_metrics.json`](artifacts/test_metrics.json) (mAP50 0.755, mAP50-95 0.636) |

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

OpenAPI UI: http://localhost:8000/docs

```bash
docker build -t runwayguard .
docker run --rm -p 8000:8000 -e MODEL_PATH=/app/weights/best.pt runwayguard
```

## Sample requests

### `GET /health`

```bash
curl -s http://localhost:8000/health
```

### `POST /detect`

```bash
curl -s -X POST http://localhost:8000/detect \
  -F "image=@sample.jpg" \
  -F "confidence=0.25"
```

```json
{
  "detections": [
    {
      "class_id": 0,
      "class_name": "fastener_hardware",
      "confidence": 0.91,
      "box": { "x1": 121.2, "y1": 87.1, "x2": 150.8, "y2": 132.4 }
    }
  ],
  "image_size": { "width": 300, "height": 300 },
  "model": "best.pt",
  "confidence_threshold": 0.25
}
```

### `POST /ask` (grounded)

```bash
curl -s -X POST http://localhost:8000/ask \
  -F "image=@sample.jpg" \
  -F "question=How many fasteners are visible?" \
  -F "confidence=0.25"
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
  "guardrail_reason": "The question asks for information that is not observable from one image."
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
| Failure IDs cited in the memo | [`artifacts/memo_failure_ids.json`](artifacts/memo_failure_ids.json) |
| Failure mining dumps | [`artifacts/failure_mine.json`](artifacts/failure_mine.json), [`artifacts/failure_small_miss.json`](artifacts/failure_small_miss.json) |
| Annotation QC | [`figures/annotation_audit/`](figures/annotation_audit/) |
| Gradio shell | [`../deploy/`](../deploy/) |
| Colab notebook | [`../notebooks/colab_train.ipynb`](../notebooks/colab_train.ipynb) |
| Hub model card | [`hf/MODEL_CARD.md`](hf/MODEL_CARD.md) |
