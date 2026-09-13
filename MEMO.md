# RunwayGuard technical memo

**Code:** https://github.com/riyaz-ahamed-07/RunwayGuard  
**Weights:** https://huggingface.co/DarkKnight1217/RunwayGuard-rtdetr-l/resolve/main/best.pt  
**sha256:** `C2D2A418069D9AC8658CA85B8359A9E1AB740CE96826C5138178B080F9243269`  
**Train:** Colab Tesla T4, Ultralytics RT-DETR-L 8.4.147, AdamW `lr0=1e-4`, batch 8, imgsz 480, seed 42  
**Logged run:** 11 epochs, ~3.9 h wall time (`docs/artifacts/results.csv`). Val mAP50-95 peaked at epoch 7 (~0.659). Ship `best.pt`, not `last.pt`.  
**Submission pack (API + links):** [`docs/SUBMISSION.md`](docs/SUBMISSION.md)

## What the system answers

Question in scope: which visible debris candidates are in this image, and where?  
Out of scope: runway reopen, Part 139 clearance, material chemistry, mass, future engine damage. FAA AC 150/5210-24 treats detection as one FOD control activity next to prevention, removal, and evaluation. One still image cannot close that loop.

## Data choices:

Source is FOD-A v2.1 only (MIT, VOC, 33,793 images at 300x300; arXiv:2110.03072). No web scrapes.

We audited the publisher ImageSets before training. Findings: duplicate list entries, IDs present in both trainval and test, and **1,403 identical dHash groups** that appear on both sides of the official split. That is a near-duplicate risk, not a proof that every collision is a video twin (plain asphalt can hash-collide). We still threw the official split out. Inflated self-report mAP is worse than a lower honest score.

Replacement split (`scripts/create_grouped_split.py`, seed 42): train 24,103 / val 4,808 / test 4,882. Groups combine dHash (threshold 18), adjacent IDs, label tuples, and environment tags. This is a heuristic. ID-disjoint manifests do not prove video independence. Test stayed frozen after weights and API thresholds were locked.

Taxonomy: 31 publisher labels collapsed many-to-one into seven operational classes in `config/taxonomy.json` (fastener_hardware, hand_tool, flexible_debris, loose_metal, plastic_paper_debris, component_container, natural_debris). The head cannot tell bolt from washer. `/ask` therefore abstains on subtype questions even when the parent class fires.

## Grouped-test numbers (frozen)

| Metric        | Value         | Read as                       | Do not read as                       |
| ------------- | ------------- | ----------------------------- | ------------------------------------ |
| mAP50         | 0.755         | Ranking quality on this split | Hidden RAP score                     |
| mAP50-95      | 0.636         | Localization under IoU sweep  | Confidence calibration               |
| Library P / R | 0.734 / 0.812 | Ultralytics COCO-style means  | Exact `/ask` behavior at 0.25 / 0.50 |

Per-class mAP50: fastener 0.821, hand_tool 0.839, flexible 0.761, loose_metal **0.541**, plastic/paper 0.966, component 0.940, natural_debris **0.418**.

Val rose from ~0.03 to ~0.80 mAP50 by epoch 4. That curve is a training diagnostic only. Test numbers above are the reported ones.

## Five failures with image IDs

Mined on the grouped test set at conf 0.25, IoU 0.5 (`scripts/mine_failures.py`, `docs/artifacts/`).

1. `016303`. Tiny `fastener_hardware` (GT area ~0.23% of 300x300). Model returned no boxes. Miss ≠ clear pavement.
2. `022564`. True `loose_metal`, plus a high-score `plastic_paper_debris` false positive. `/ask` answers only use scores ≥0.50; evidence still collects at 0.25 so mid-band boxes are not hidden from the policy layer.
3. `027929`. GT `natural_debris` matched as `loose_metal`, with an extra plastic box. Wrong ontology; a human may still want to inspect the same spot.
4. `027930`. Same bias in cleaner form: rock/wood labeled only as `loose_metal`. Matches the weak class AP (0.418).
5. `022573`. True `loose_metal` with a high-conf `natural_debris` false positive. Fits the class P/R asymmetry (P 0.492, R 0.968).

## Part B (no agent stack)

Intent parsing and arithmetic live in `app/reasoning.py`. Optional OpenAI HTTP can propose a constrained intent JSON (`app/llm_phrase.py`); the spoken answer is always rendered in code. Docker sets `RUNWAYGUARD_LLM=0`. No LangChain, LangGraph, CrewAI, or AutoGen.

Examples: "Is the runway safe to reopen?" → `UNOBSERVABLE` / `INSUFFICIENT_INFORMATION`. "Is there a screwdriver?" with `hand_tool`@0.93 → abstain; the class is broader than the noun.

## What is ready vs what is not claimed

Ready: FastAPI `/health` `/detect` `/ask`, Docker, request logging, upload limits, shared imgsz 480, pytest suite, HF weights, optional Gradio shell under `deploy/`.

Not claimed: calibrated probabilities, API P/R equal to Ultralytics library means without a separate operating-point table, leakage-free video identity, or any replacement for authorized inspection.
