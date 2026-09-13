# RunwayGuard

### Technical memo · Constrained object detection & evidence-grounded reasoning

> **Detect visible debris. Explain the evidence. Never certify a clear runway.**

[Source code](https://github.com/riyaz-ahamed-07/RunwayGuard) · [Trained weights](https://huggingface.co/DarkKnight1217/RunwayGuard-rtdetr-l/resolve/main/best.pt) · [Setup & API examples](docs/SUBMISSION.md) · [Test metrics](docs/artifacts/test_metrics.json)

## 1 · Problem and engineering decisions

RunwayGuard identifies **visible foreign object debris (FOD) candidates in a still image**, then answers questions using structured detections. Its intended role is inspection support, not runway reopening authorization. The workflow is evidence extraction, explicit rules, and human review of consequential decisions. Airport-system integration remains future work.

| Decision | Rationale and cost |
| --- | --- |
| Fine-tune RT-DETR-L on FOD-A | Domain-specific labels beyond COCO; benchmark performance is not proof of airport readiness. |
| Merge 31 labels into seven classes | Broad categories trade subtype detail for a coarser task; a hand-tool detection cannot establish “screwdriver.” |
| Separate evidence from answer policy | Claims remain inspectable; deterministic guardrails cannot repair incorrect detections. |

## 2 · Data and reproducibility

**FOD-A v2.1:** 33,793 images at 300 × 300; Pascal VOC annotations; no additional web-scraped training data. Evidence: [dataset paper](https://arxiv.org/abs/2110.03072), [taxonomy](config/taxonomy.json), [annotation-audit contact sheet](docs/figures/annotation_audit/annotation_audit_01.jpg). The contact sheet shows annotation checks, **not predictions**.

The publisher-split audit found duplicate entries, overlapping IDs, and **1,403 identical dHash groups spanning trainval/test**. Hash collisions indicate contamination risk, not proven duplicate sequences. The replacement [grouping script](scripts/create_grouped_split.py) combines dHash distance (threshold 18), adjacent IDs, label tuples, and environment tags. Seed 42 yields **24,103 train / 4,808 validation / 4,882 test** images. This heuristic does **not** establish video independence or eliminate leakage. Reported test results use the frozen checkpoint and thresholds; failure inspection is post-evaluation diagnosis, not a tuning set.

**Training:** Colab Tesla T4; Ultralytics 8.4.147; pretrained RT-DETR-L; AdamW, `lr0=1e-4`, batch 8, `imgsz=480`, seed 42. The [log](docs/artifacts/results.csv) records **11 epochs, ~3.9 hours**, not a completed 25-epoch run. Validation mAP50–95 peaked at **0.65885 in epoch 7**; the delivered checkpoint is `best.pt`.

## 3 · Results and limitations

| Grouped-test metric | Result | Interpretation |
| --- | ---: | --- |
| mAP50 | **0.755** | Detection AP at IoU 0.50 |
| mAP50–95 | **0.636** | AP averaged across stricter localization thresholds |
| Library precision / recall | **0.734 / 0.812** | Not measured API P/R at 0.25 / 0.50 |

**Per-class mAP50:** fastener_hardware 0.821; hand_tool 0.839; flexible_debris 0.761; **loose_metal 0.541**; plastic_paper_debris 0.966; component_container 0.940; **natural_debris 0.418**. These are saved grouped-test results, not hidden-test scores. Weak natural/metal separation and tiny-object misses limit usefulness; confidence is not calibrated probability.

## 4 · Five failures → targeted next experiments

Mined at **confidence 0.25, IoU 0.50**. IDs open **original images without prediction overlays**; observations come from the [failure records](docs/artifacts/failure_mine.json) and [tiny-object pass](docs/artifacts/failure_small_miss.json). Causes are hypotheses, not proven mechanisms. Similar adjacent frames are separate cases, **not five independent failure modes**.

| Image | Recorded failure | Likely cause → next experiment |
| --- | --- | --- |
| [016303](deploy/samples/016303.jpg) | Fastener covers ~0.23% of image; no boxes. | Limited detail against textured pavement → measure size-binned recall; test validation crops/tiling. Upscaling cannot recover missing source detail. |
| [022564](deploy/samples/022564.jpg) | Loose-metal ground truth; extra plastic/paper prediction. | Small shape and pavement texture may produce competing responses → inspect box overlap and annotation completeness before adding hard negatives. |
| [027929](deploy/samples/027929.jpg) | Natural debris predicted as metal, plus plastic/paper. | Small, similarly colored material cues may be ambiguous → inspect matched boxes; test balanced natural/metal training examples. |
| [027930](deploy/samples/027930.jpg) | Natural debris predicted only as metal. | Near-identical appearance to 027929 suggests repeatable confusion → validate on distinct capture groups, not more neighboring frames. |
| [022573](deploy/samples/022573.jpg) | Loose-metal ground truth; extra natural-debris prediction. | Similar scene to 022564 but different extra class suggests unstable separation → inspect per-box scores/overlap and measure duplicate-count errors. |

**Next priority:** tiny-object recall and false candidate counts, not more features. These are proposed validation experiments, not achieved improvements.

## 5 · Routing and guardrails

[Local routing](app/reasoning.py) distinguishes detection questions, questions needing no detection, and unobservable requests. `/ask` collects evidence at **0.25**, regardless of caller threshold; affirmative answer evidence uses **≥0.50**. Counts describe predicted candidates, not guaranteed physical-object totals. No detections never establishes clear pavement.

Optional [direct OpenAI HTTP](app/llm_phrase.py) proposes intent JSON. Code validates operations, categories and evidence IDs, preserves protected local decisions, and renders the answer. Invalid proposals fall back to deterministic handling. **The LLM does not freely author answers or determine runway safety.** No agent framework is used; the deterministic path needs no provider key.

**Insufficient-information examples:** “Is the runway safe to reopen?” → `UNOBSERVABLE` / `INSUFFICIENT_INFORMATION`. “Is there a screwdriver?” with `hand_tool` at 0.93 → abstain: the trained category cannot resolve that subtype.

## 6 · Delivery and verification boundary

Local review verified checkpoint loading, real-image `/detect` and `/ask` HTTP 200 responses, and **42 passing tests at the reviewed revision**. Public weights were accessible without authentication. Docker/demo assets are supplied; live provider execution, Docker execution and airport deployment are **not claimed verified**. Subsequent code changes require retesting. See the [submission guide](docs/SUBMISSION.md) for setup and API examples.

**Checkpoint SHA256** ([artifact record](docs/artifacts/test_metrics.json)):

`C2D2A418069D9AC8658CA85B8359A9E1AB740CE96826C5138178B080F9243269`
