# RAP submission pack

This page is the checklist for reviewers. The written memo stays in [`MEMO.md`](../MEMO.md). Everything else required by the brief is linked here.

## Four required deliverables

| # | Requirement | Location |
| --- | --- | --- |
| 1 | GitHub repo (train, eval, inference, API) | https://github.com/riyaz-ahamed-07/RunwayGuard |
| 2 | Trained weights (direct download) | https://huggingface.co/DarkKnight1217/RunwayGuard-rtdetr-l/resolve/main/best.pt |
| 3 | Written memo (≤2 pages) | [`MEMO.md`](../MEMO.md) |
| 4 | API instructions + sample payloads | This page (below) and [`README.md`](../README.md) |

### Weights card

| Item | Value |
| --- | --- |
| Hub repo | https://huggingface.co/DarkKnight1217/RunwayGuard-rtdetr-l |
| File | `best.pt` |
| sha256 | `C2D2A418069D9AC8658CA85B8359A9E1AB740CE96826C5138178B080F9243269` |
| Train record | `docs/artifacts/results.csv` (11 epochs logged, ~3.9 h on Colab T4) |
| Test metrics | `docs/artifacts/test_metrics.json` (grouped test: mAP50 0.755, mAP50-95 0.636) |

```bash
hf download DarkKnight1217/RunwayGuard-rtdetr-l best.pt --local-dir weights
# PowerShell: Get-FileHash weights\best.pt -Algorithm SHA256
```

---

## Quick start (reviewer)

```bash
git clone https://github.com/riyaz-ahamed-07/RunwayGuard.git
cd RunwayGuard
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
hf download DarkKnight1217/RunwayGuard-rtdetr-l best.pt --local-dir weights
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

OpenAPI UI: http://localhost:8000/docs

Docker (after `weights/best.pt` exists):

```bash
docker build -t runwayguard .
docker run --rm -p 8000:8000 -e MODEL_PATH=/app/weights/best.pt runwayguard
```

---

## API samples

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

Response shape:

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

### `POST /ask` (grounded count)

```bash
curl -s -X POST http://localhost:8000/ask \
  -F "image=@sample.jpg" \
  -F "question=How many fasteners are visible?" \
  -F "confidence=0.25"
```

### `POST /ask` (unobservable / abstain)

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

**Policy notes**

- `/ask` evidence collection is fixed at confidence **0.25**.
- Answered claims use scores ≥ **0.50**.
- Shared inference size: **imgsz 480**.
- Subtype questions (e.g. “screwdriver?”) abstain even if `hand_tool` fires.
- Optional LLM path proposes structured intent JSON only; final text is code-rendered. Docker sets `RUNWAYGUARD_LLM=0`.

---

## Extra material (not required by the brief)

| Extra | Path |
| --- | --- |
| Failure mining IDs used in the memo | [`artifacts/memo_failure_ids.json`](artifacts/memo_failure_ids.json) |
| Full mine dumps | [`artifacts/failure_mine.json`](artifacts/failure_mine.json), [`failure_small_miss.json`](artifacts/failure_small_miss.json) |
| Annotation QC figures | [`figures/annotation_audit/`](figures/annotation_audit/) |
| Gradio demo shell | [`../deploy/`](../deploy/) |
| Colab train notebook | [`../notebooks/colab_train.ipynb`](../notebooks/colab_train.ipynb) |
| Hub model card source | [`hf/MODEL_CARD.md`](hf/MODEL_CARD.md) |

---

## SharePoint drop (suggested)

Upload a short note that points here. Do not upload `best.pt` or the dataset.

```text
Repo:     https://github.com/riyaz-ahamed-07/RunwayGuard
Memo:     https://github.com/riyaz-ahamed-07/RunwayGuard/blob/main/MEMO.md
Pack:     https://github.com/riyaz-ahamed-07/RunwayGuard/blob/main/docs/SUBMISSION.md
Weights:  https://huggingface.co/DarkKnight1217/RunwayGuard-rtdetr-l
```
