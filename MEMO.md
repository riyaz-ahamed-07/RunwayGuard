# RunwayGuard — Technical Memo

**Repo:** https://github.com/riyaz-ahamed-07/RunwayGuard  
**Stack:** Ultralytics RT-DETR-L (v8.4.147) · FastAPI `/detect` + `/ask` · Docker  
**Train hardware:** Google Colab Tesla T4 · seed 42 · AdamW lr `1e-4` · imgsz 480 · batch 8  
**Run record:** epochs completed ____ · wall-clock ____ · weights: ____ *(public URL → `weights/best.pt`)*

---

## Why this problem (and what it is not)

Airport **foreign object debris (FOD)** can damage aircraft on the AOA; FAA FOD programs treat **detection** as one pillar beside prevention, removal, and evaluation—not as a substitute for authorized inspection (AC 150/5210-24). RunwayGuard therefore answers a narrow, observable question: *what visible debris candidates appear in this image, and where?* It does **not** certify that a runway is clear, safe to reopen, or free of unobserved hazards.

I chose FOD-A over saturated tutorial domains because: (1) every target class is **non-COCO**; (2) the data is MIT-licensed, peer-reviewed ([arXiv:2110.03072](https://arxiv.org/abs/2110.03072)), and operationally meaningful; (3) Part B has a sharp abstention boundary that mirrors real automation policy—detect candidates, escalate to humans. Early alternatives (other safety niches) were dropped once FOD-A’s license, label density, and reasoning story were clear.

**Sourcing.** FOD-A v2.1 Pascal VOC only (33,793 images, 300×300, XML boxes). No scraping, no silent relabeling. I validated XML parseability and box geometry, spot-checked stratified overlays in `docs/figures/annotation_audit/`, then mapped 31 publisher labels → **7 operational classes** in `config/taxonomy.json` (inspect/remove response is shared within a class; several fine labels lacked enough independent capture groups for leakage-free per-class AP if kept separate). Annotations were merged, not discarded.

## Split strategy (why the official lists were rejected)

Video-derived detection sets inflate mAP when near-duplicate frames land in both train and test. Auditing the publisher ImageSets showed duplicate IDs, train/test overlap, and **1,403** identical, label-compatible perceptual-hash groups crossing the supplied split. Using that split would have produced a metric I could not defend.

**Fix:** deterministic **grouped split** (`scripts/create_grouped_split.py`, seed **42**, ~70/15/15) that keeps related captures together—**train 24,103 / val 4,808 / test 4,882** (`data_splits/`). The grouped test set stays frozen until weights and thresholds are fixed. That is the evaluation I trust; RAP’s hidden set remains the external judge.

## Metrics — reading them like an engineer

| Number | Decision it supports | What it does *not* support |
|--------|----------------------|----------------------------|
| mAP50 | “Are boxes roughly on the object?” | Tight localization / ops clearance |
| mAP50–95 | Localization under stricter IoU | Cross-airport camera shift |
| P / R | False alarms vs misses at a threshold | “No box ⇒ runway clear” |
| Per-class AP / confusion | Which operational classes collide | Material, mass, or future damage |

**Self-reported grouped-test results** *(fill from `evaluate.py` after freeze—do not substitute validation curves)*:  
mAP50 ____ · mAP50–95 ____ · P ____ · R ____  

*Training diagnostic only:* on T4 validation, mAP50 moved from ~0.03 (epoch 1) to ~0.80 by epoch 4 while losses fell—evidence of learning, **not** the score claimed above.

Self-reported test mAP is secondary to the hidden set. A lower, leakage-aware number with named failure modes is preferable to an inflated official-split score.

## Five failure modes (structured error analysis)

RAP grades root-cause honesty. Failures are organized as a **competence map** (mode → evidence → disposition), not as “the model is bad.” Replace each `⟦id⟧` with a real grouped-test example after error mining; do not invent IDs.

1. **Small-object miss.** FOD-A contains thousands of boxes &lt;1% of image area; RT-DETR at 480 px still drops tiny fasteners. *Evidence:* `⟦id⟧`. *Disposition:* keep high-recall screening threshold; never treat a miss as clearance.
2. **Background / marking false positive.** Wet sheen, paint edges, or joints mimic debris texture. *Evidence:* `⟦id⟧`. *Disposition:* Part B requires ≥0.50 for answers; human review on low-confidence candidates.
3. **Taxonomy confusion.** Operational merge (e.g. metal vs fastener hardware) trades fine ID for stable AP; visually adjacent classes still swap. *Evidence:* `⟦id⟧`. *Disposition:* acceptable if both classes trigger the same inspect/remove action; report confusion explicitly.
4. **Condition shift (blur / compression / lighting).** Motion blur and harsh illumination degrade edges on 300×300 sources. *Evidence:* `⟦id⟧`. *Disposition:* document as in-distribution weakness; prioritize clearer captures operationally.
5. **Closed-set novelty.** Debris outside the 31-label taxonomy is forced into nearest class or missed. *Evidence:* `⟦id⟧`. *Disposition:* out of scope for this closed detector; abstain rather than invent a class.

## Part B — when the detector is called (and when it must refuse)

Hand-written router in `app/reasoning.py` (no LangChain / CrewAI / AutoGen):

1. **UNOBSERVABLE** (safe to reopen / clear to land / damage / weight / ownership…) → `INSUFFICIENT_INFORMATION`; detector not required to refuse a safety claim.  
2. **NO_DETECTION_NEEDED** (off-topic) → abstain.  
3. **DETECT** → run RT-DETR; answer only from detections ≥ **0.50** (form `confidence` default **0.25** only filters returned boxes). Non-detection never becomes “absent” or “clear.”

**Insufficient-information example.**  
Q: *“Is the runway safe to reopen?”* → route `UNOBSERVABLE`, status `INSUFFICIENT_INFORMATION`: one image can support visible candidates only; authorized inspection remains required.  
Same policy for *“Is the runway clear?”* and for class questions with no confident hit (“not proof that none is present”).

## Seams (scope honesty)

Production-shaped today: upload limits, request IDs, structured logging, Docker, unit + API tests, reproducible split/train scripts. Explicitly **not** claimed: multi-airport calibration, continuous airfield sensors, or replacement of Part 139 inspection. With more time: size-stratified AP, condition-tagged error slices, and a longer train under the same seed.

---

*Before SharePoint: paste weights URL + test metrics + five `⟦id⟧` evidence lines, export ≤2 pages PDF.*
