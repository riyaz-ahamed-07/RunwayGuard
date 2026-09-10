---
license: mit
library_name: ultralytics
tags:
  - object-detection
  - rtdetr
  - fod
  - airport
  - runwayguard
pipeline_tag: object-detection
---

# RunwayGuard RT-DETR-L (FOD-A grouped split)

Fine-tuned **RT-DETR-L** (Ultralytics 8.4.147) for **visible airport foreign-object debris (FOD)** on FOD-A v2.1, using a deterministic **grouped** train/val/test split (seed 42).

## Files

| File | Role |
|------|------|
| `best.pt` | Selected checkpoint (best val fitness during training) |
| `results.csv` | Per-epoch train/val curves |
| `test_metrics.json` | Held-out grouped-test metrics |

## Checksum

```text
SHA256(best.pt) = C2D2A418069D9AC8658CA85B8359A9E1AB740CE96826C5138178B080F9243269
```

## Grouped-test (frozen)

| Metric | Value |
|--------|-------|
| mAP50 | 0.755 |
| mAP50–95 | 0.636 |
| Library P / R | 0.734 / 0.812 |

Seven operational classes: `fastener_hardware`, `hand_tool`, `flexible_debris`, `loose_metal`, `plastic_paper_debris`, `component_container`, `natural_debris`.

## Usage

```bash
# download
huggingface-cli download USER/RunwayGuard-rtdetr-l best.pt --local-dir ./weights

# or
curl -L -o weights/best.pt https://huggingface.co/USER/RunwayGuard-rtdetr-l/resolve/main/best.pt
```

Point the FastAPI app at the local file:

```bash
export MODEL_PATH=weights/best.pt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Scope / limits

Detects **visible candidate debris** in an image. Does **not** certify runway clearance, material, mass, or future aircraft damage. See the GitHub repo memo for split rationale and failure cases.

**Code:** https://github.com/riyaz-ahamed-07/RunwayGuard  
**Data:** FOD-A (MIT) — cite [arXiv:2110.03072](https://arxiv.org/abs/2110.03072)
