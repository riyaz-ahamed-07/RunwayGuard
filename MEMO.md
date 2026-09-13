# RunwayGuard — Technical Memo (≤2 pages)

**Candidate repo:** https://github.com/riyaz-ahamed-07/RunwayGuard  
**Model:** Ultralytics RT-DETR-L fine-tuned on FOD-A (7 operational classes)  
**Hardware / time:** *[fill after run — e.g. Colab Tesla T4, N epochs, wall-clock hours]*  
**Weights:** *[public download URL]* → load as `weights/best.pt` or `MODEL_PATH`

---

## 1. Domain, dataset, and sourcing

**Problem.** Detect *visible* airport foreign-object debris (FOD) in runway/taxiway imagery to prioritize inspection. FOD can damage aircraft; clearance remains a human / operational decision.

**Why this domain.** Life-safety framing; non-COCO debris classes; strong fit for a constrained reasoning API (“I see candidates” vs “runway is safe to reopen”). Avoids saturated peer themes (generic PPE-only tutorials) while staying auditable.

**Data.** Official **FOD-A v2.1** Pascal VOC ([FOD-UNOmaha/FOD-data](https://github.com/FOD-UNOmaha/FOD-data), MIT; [arXiv:2110.03072](https://arxiv.org/abs/2110.03072)). 33,793 images at 300×300 with XML boxes. Labels were **not** re-scraped; I audited structure (parseable XML, valid boxes) and produced stratified boxed samples under `docs/figures/annotation_audit/`.

**Taxonomy.** 31 source labels map into seven operational classes (`config/taxonomy.json`): fastener hardware, hand tools, flexible debris, loose metal, plastic/paper, component/container, natural debris. Fine labels that do not change the operational response (inspect/remove) and that lack enough independent capture groups for leakage-free AP were merged deliberately—not dropped.

**Mid-project note.** Early exploration considered other life-safety domains; FOD-A was selected for official licensing, dense annotations, and a clear Part B abstention story.

---

## 2. Train / val / test split

The **publisher’s ImageSets split was not used**. Audit findings included duplicate list entries, IDs in both trainval and test, and many near-duplicate / identical perceptual-hash groups spanning the official split. That would leak near-identical frames into evaluation and inflate self-reported mAP.

**Replacement:** deterministic **grouped split** (`scripts/create_grouped_split.py`, seed **42**, ~70/15/15) that keeps related captures together: **train 24,103 / val 4,808 / test 4,882**. Manifests live in `data_splits/`. The grouped **test** set is frozen until model and thresholds are fixed.

---

## 3. Metrics — what they tell you (and what they don’t)

| Metric | Role |
|--------|------|
| mAP50 / mAP50-95 | Ranking quality of boxes vs GT on *this* distribution |
| Precision / recall | False alarms vs misses at the operating threshold |
| Per-class AP / confusion | Which operational classes collide (e.g. metal vs fastener) |

**Self-reported (grouped test) — fill after `evaluate.py`:**

- mAP50: *____* mAP50-95: *____* P: *____* R: *____*
- Hyperparameters: RT-DETR-L, imgsz ___, batch ___, epochs completed ___, seed 42, Ultralytics 8.4.147

**What they do not tell you.** They are **not** RAP’s hidden-set score; they do **not** prove operational runway clearance; staged 300×300 FOD-A may shift under different airports/cameras; non-detection ≠ “clear.”

---

## 4. Five failure cases (observed — fill after training)

*Do not invent. After test inference, replace each row with a real image id, screenshot path, and cause.*

| # | Image / path | Symptom | Root cause |
|---|--------------|---------|------------|
| 1 | *[id]* | Miss / FP / wrong class | e.g. tiny object (&lt;1% area) |
| 2 | *[id]* | | e.g. blur / compression |
| 3 | *[id]* | | e.g. wet reflection / line edge FP |
| 4 | *[id]* | | e.g. class confusion within taxonomy |
| 5 | *[id]* | | e.g. lighting / glare / novel debris |

---

## 5. Part B reasoning — detector vs abstain

Hand-written router in `app/reasoning.py` (no LangChain/CrewAI/etc.):

1. **UNOBSERVABLE** (safe-to-reopen, damage prediction, weight, ownership, …) → abstain; **detector not required** for the safety claim.
2. **NO_DETECTION_NEEDED** (off-topic) → abstain.
3. **DETECT** → run RT-DETR; answer only from boxes/classes with confidence ≥ **0.50**; otherwise **INSUFFICIENT_INFORMATION**. Absence is never certified as “clear.”

**Concrete insufficient-information example**

- **Q:** “Is the runway safe to reopen?”  
- **Route:** `UNOBSERVABLE` **Status:** `INSUFFICIENT_INFORMATION`  
- **Why:** One image cannot certify operational safety; the API refuses to guess.

A second pattern: class-specific “is X present?” with no confident detection → insufficient information, not “X is absent.”
