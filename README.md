# RunwayGuard

Constrained **RT-DETR** object detection + FastAPI reasoning for **visible airport foreign-object debris (FOD)**.

Built for RAP Pre-Hackathon Screening (Round 1): Part A detection API, Part B structured intent + optional direct LLM phrasing (no agent frameworks), reproducible training, honest evaluation.

| Endpoint | Role |
|----------|------|
| `GET /health` | Model readiness |
| `POST /detect` | Classes, boxes, confidences |
| `POST /ask` | Intent routing → structured reasoning → confidence / abstain |

**Scope:** inspection-prioritization aid for *visible candidate debris*. It does **not** certify a runway as safe, replace authorized inspection, or invent unobserved facts (material, weight, future damage).

---

## Repository layout

```text
RunwayGuard/
├── app/                      # FastAPI (/detect, /ask, /health)
├── config/taxonomy.json      # 31 FOD-A labels → 7 operational classes
├── data_splits/              # Grouped train/val/test manifests (seed 42)
├── docs/
│   ├── figures/annotation_audit/   # Label QC samples
│   └── ANNOTATION_AUDIT.md
├── notebooks/colab_train.ipynb     # GPU training notebook
├── scripts/                  # Split, VOC→YOLO, audit render
├── tests/                    # Reasoning unit tests
├── weights/                  # Place best.pt here (gitignored)
├── train.py                  # RT-DETR fine-tune
├── evaluate.py               # Held-out test metrics
├── MEMO.md                   # ≤2-page written memo (deliverable)
├── Dockerfile
└── requirements.txt
```

---

## Why this domain and dataset

| Choice | Rationale |
|--------|-----------|
| Problem | Airport FOD is life-safety relevant; detection maps cleanly to boxes/classes |
| Dataset | [FOD-A v2.1](https://github.com/FOD-UNOmaha/FOD-data) (MIT), Pascal VOC, peer-reviewed ([arXiv:2110.03072](https://arxiv.org/abs/2110.03072)) |
| Non-COCO | All seven target classes are operational FOD categories, not stock COCO labels |
| Framing | Detector finds *visible candidates*; clearance stays a human decision (RAP-style guardrail) |

FAA FOD context: <https://www.faa.gov/airports/airport_safety/fod>

---

## Dataset and split (audit-first)

**Source:** FOD-A v2.1 Pascal VOC — 33,793 images (300×300), 31 source labels, XML boxes.

**Official split was rejected.** Audit of publisher ImageSets found duplicate list entries, IDs in both trainval and test, and **1,403 identical dHash groups** that appeared on both sides of the official split (proxy near-duplicate risk; not all are proven label-compatible video twins, and plain backgrounds can collide). Keeping that split would have risked inflated self-reported mAP.

**Replacement:** deterministic **grouped split** (`scripts/create_grouped_split.py`, seed **42**) using dHash + adjacent IDs + label/environment tags as **heuristic** groups—not certified video identities:

| Split | Images |
|-------|--------|
| train | 24,103 |
| val | 4,808 |
| test | 4,882 |

Manifests are pairwise ID-disjoint (`data_splits/`). Zero matching hashes under the rule does **not** prove all leakage is eliminated.

**Taxonomy:** 31 source labels map **many-to-one** into seven **operational grouping** classes in `config/taxonomy.json` (not material facts; e.g. `paint chip` ≠ proven plastic composition). Every source label is retained exactly once:

`fastener_hardware`, `hand_tool`, `flexible_debris`, `loose_metal`, `plastic_paper_debris`, `component_container`, `natural_debris`

Annotation QC figures: `docs/figures/annotation_audit/`.

---

## Constraints checklist

| Constraint | Status |
|------------|--------|
| RT-DETR fine-tune (Ultralytics) | Yes — `train.py` |
| ≥1 non-COCO class | Yes — all 7 |
| No LangChain / LangGraph / CrewAI / AutoGen | Yes — deterministic intent/policy in `app/reasoning.py`; optional OpenAI HTTP phrasing only |
| No AutoML / no-code training | Yes |
| Reproducible prep + train + eval | Yes — this README + notebook |
| Docker | Yes — `Dockerfile` |

---

## Reproduce data preparation

1. Download **FOD-A v2.1 Pascal VOC** from the official repo (Drive file id used in the notebook: `1RdErcq8PGRXZUOGauaACkQG44T-QyZ4x`).
2. Point at the folder that contains `Annotations`, `JPEGImages`, `ImageSets` (usually `VOC2007`).

```bash
python scripts/create_grouped_split.py --dataset-root /path/to/VOC2007 --output-dir data_splits

python scripts/convert_voc_to_yolo.py \
  --dataset-root /path/to/VOC2007 \
  --splits-dir data_splits \
  --taxonomy config/taxonomy.json \
  --output-dir prepared_data \
  --clean-output
```

Use `--clean-output` whenever regenerating so stale files cannot contaminate Ultralytics directory loaders.
`prepared_data/` is local-only (gitignored). Colab rebuilds it from FOD-A + committed manifests.

---

## Training (reproducibility)

**Recommended path:** GPU Colab / cloud. Local GTX-class boxes may be VRAM-limited; system PyTorch must be CUDA-enabled.

```bash
# Smoke: 1 epoch on 2% of train
python train.py --data prepared_data/data.yaml --smoke-test

# Baseline (document actual wall time + hardware in MEMO.md after the run)
python train.py \
  --data prepared_data/data.yaml \
  --model rtdetr-l.pt \
  --epochs 25 \
  --imgsz 480 \
  --batch 8 \
  --device 0 \
  --seed 42 \
  --project runs \
  --name rtdetr_l_foda
```

| Item | Value |
|------|--------|
| Library | Ultralytics `8.4.147` (`requirements.txt`) |
| Architecture | RT-DETR-L, COCO-pretrained backbone/head remapped to `nc=7` |
| Optimizer | AdamW, `lr0=1e-4`, `weight_decay=1e-4` |
| Seed | 42 (`deterministic=False` — CUDA deformable-attention backward is not bit-exact) |
| Recorded env | `runs/.../environment.json` written by `train.py` |

**Time-limited run (optional):** fewer epochs and/or `imgsz 416` — state the exact command and stop epoch in `MEMO.md`. Prefer a finished shorter run over an unfinished 25-epoch plan.

Notebook: `notebooks/colab_train.ipynb`.

---

## Evaluation

Do **not** tune on the grouped test set. After freezing weights:

```bash
python evaluate.py \
  --weights weights/best.pt \
  --data prepared_data/data.yaml \
  --imgsz 480 \
  --batch 8 \
  --device 0 \
  --output runs/test_metrics.json
```

Report mAP50, mAP50-95, precision/recall, per-class AP / confusion behavior, and **five observed** failure cases in `MEMO.md`. Self-reported metrics are secondary to RAP’s hidden set; inventing failures or scores is worse than admitting limits.

---

## Model weights

`MODEL_PATH` must be a **local filesystem path** to a `.pt` file (not a URL). Download first, then point:

```bash
# Example after you publish a Drive/Release link:
# curl -L -o weights/best.pt "YOUR_DIRECT_DOWNLOAD_URL"
# sha256sum weights/best.pt   # record checksum in MEMO.md
export MODEL_PATH=weights/best.pt   # optional; default is weights/best.pt
```

**Reviewer download:** *[paste working public URL + sha256 after training]*

Do not commit multi‑hundred‑MB `.pt` files to git.

---

## Run the API

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt

# weights/best.pt must exist (or MODEL_PATH set)
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

OpenAPI docs: <http://localhost:8000/docs>

### `POST /detect`

```bash
curl -s -X POST http://localhost:8000/detect \
  -F "image=@sample.jpg" \
  -F "confidence=0.25"
```

Example response **shape** (illustrative schema, not a claimed prediction):

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

### `POST /ask`

```bash
curl -s -X POST http://localhost:8000/ask \
  -F "image=@sample.jpg" \
  -F "question=How many bolts are visible?" \
  -F "confidence=0.25"
```

**Insufficient-information example** (safety certification — no detector call for unobservable claims):

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

### Part B policy

1. **Deterministic intent grammar + arithmetic** in `app/reasoning.py` (authoritative answers).
2. **Optional direct OpenAI HTTP** may propose **structured intent JSON only** (`app/llm_phrase.py`). Proposals are schema-validated (allowed kinds, categories, evidence IDs). The final sentence is always rendered in code. Free-form model text never becomes the answer. Set `OPENAI_API_KEY` to exercise the path; `RUNWAYGUARD_LLM=0` forces deterministic-only (default in Docker).
3. **`/ask` evidence confidence is fixed at 0.25**; answers use ≥ **0.50**. `/detect` keeps its own display filter.
4. Shared **imgsz=480**. Subtype / unknown-target / negation / exclusion / spatial / colour questions abstain.

```bash
pytest -q
```

---

## Docker

```bash
# After weights/best.pt exists:
docker build -t runwayguard .
docker run --rm -p 8000:8000 -e MODEL_PATH=/app/weights/best.pt runwayguard
```

---

## Deliverables map (RAP)

| Deliverable | Location |
|-------------|----------|
| Source (train, eval, FastAPI) | this repo |
| Weights link | `MEMO.md` + README weights section |
| Written memo (≤2 pages) | [`MEMO.md`](MEMO.md) |
| API usage + payloads | this README |
| Bonus Docker / logging / errors | `Dockerfile`, request logging + HTTP errors in `app/main.py` |

---

## License

Code in this repository is provided for the RAP screening submission. FOD-A data remains under its upstream MIT license; cite the FOD-A authors when redistributing derived artifacts.
