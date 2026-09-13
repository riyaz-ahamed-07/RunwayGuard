# RunwayGuard — Technical Memo

**Repo:** https://github.com/riyaz-ahamed-07/RunwayGuard  
**Stack:** Ultralytics RT-DETR-L (v8.4.147) · FastAPI `/detect` + `/ask` · Docker · imgsz 480  
**Train hardware:** Google Colab Tesla T4 · seed 42 · AdamW lr `1e-4` · batch 8  
**Run record:** epochs completed ____ · wall-clock ____ · weights URL ____ · sha256 ____  

---

## Why this problem (and what it is not)

Airport **foreign object debris (FOD)** can damage aircraft on the AOA; FAA FOD programs treat **detection** as one pillar beside prevention, removal, and evaluation—not as a substitute for authorized inspection (AC 150/5210-24). RunwayGuard answers: *what visible debris candidates appear, and where?* It does **not** certify clearance, reopen safety, material, mass, or future damage.

**Sourcing.** FOD-A v2.1 Pascal VOC only (33,793×300², MIT; [arXiv:2110.03072](https://arxiv.org/abs/2110.03072)). No scraping. Spot-checked overlays in `docs/figures/annotation_audit/`. Mapped 31 publisher labels **many-to-one** into seven **operational groupings** (`config/taxonomy.json`)—inspect/remove response shared within a class; subtypes are not independently identifiable at the head. Merging does not create new independent captures for rare subtypes.

## Split strategy

Publisher ImageSets showed duplicate IDs, train/test overlap, and **1,403 identical dHash groups** spanning the official split (near-duplicate **risk**; not all proven label-matched video twins; plain backgrounds can collide). That split was **not** used.

**Replacement:** deterministic grouped split (seed **42**): train **24,103** / val **4,808** / test **4,882**. Grouping uses 64-bit dHash, adjacent filenames, label tuple, and environment tags (threshold 18)—a **heuristic**, not verified shot IDs. Manifests are ID-disjoint; that does **not** prove full independence. Test stays frozen until weights/thresholds are fixed.

## Metrics

| Number | Supports | Does not support |
|--------|----------|------------------|
| Ultralytics mAP50 / mAP50–95 | Box ranking on this distribution | Ops clearance; hidden RAP set |
| Library mean P/R | Aggregate COCO-style summary | Exact API ops at 0.25/0.50 without a separate operating-point eval |
| Per-class AP | Class collisions under the 7-way ontology | Subtype identity (bolt vs washer) |

**Grouped-test (fill after freeze):** mAP50 ____ · mAP50–95 ____ · library P/R ____  

*Training diagnostic only (val, not test):* mAP50 ~0.03→~0.80 by epoch 4 on T4—learning signal only.

## Five failure modes (mine real IDs after test)

1. **Small-object miss** — many GT boxes &lt;1% area. Evidence `⟦id⟧`. Never treat miss as clear.  
2. **Marking/reflection FP** — paint edges, wet sheen. Evidence `⟦id⟧`. Answer threshold 0.50.  
3. **Operational-class confusion** — e.g. metal vs fastener. Evidence `⟦id⟧`. Same inspect action may still hold.  
4. **Blur / lighting / compression** — Evidence `⟦id⟧`.  
5. **Closed-set limit** — novel debris forced to nearest class or missed. Evidence `⟦id⟧` only if observed in-distribution failure; do not invent OOD examples.

## Part B

Deterministic intent/policy in `app/reasoning.py`; optional OpenAI HTTP **phrasing** of an already-validated conclusion (`OPENAI_API_KEY`, `RUNWAYGUARD_LLM=0` to disable). No agent frameworks. `/ask` always collects evidence at **0.25** (caller cannot hide mid-scores); answers use ≥**0.50**. Subtype questions (“screwdriver?”, “how many bolts?”) abstain even if the parent class fires. Category plurals count only that class.

**Example.** Q: “Is the runway safe to reopen?” → `UNOBSERVABLE` / `INSUFFICIENT_INFORMATION`.  
**Example.** Q: “Is there a screwdriver?” + `hand_tool`@0.93 → insufficient: class is broader than screwdriver.

## Seams

Ready: logging, upload limits, Docker, tests, resume-fails-if-missing, shared imgsz. Not claimed: Part 139 replacement, calibrated confidence, proven leakage-free video splits.

---

*Fill weights URL, test metrics, and five observed IDs before SharePoint; keep ≤2 pages.*
