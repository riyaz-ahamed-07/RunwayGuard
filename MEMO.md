# RunwayGuard technical memo

<table>
<tr><td><b>Code</b></td><td><a href="https://github.com/riyaz-ahamed-07/RunwayGuard">github.com/riyaz-ahamed-07/RunwayGuard</a></td></tr>
<tr><td><b>Weights</b></td><td><a href="https://huggingface.co/DarkKnight1217/RunwayGuard-rtdetr-l">DarkKnight1217/RunwayGuard-rtdetr-l</a> (<code>best.pt</code>)</td></tr>
<tr><td><b>Direct file</b></td><td><a href="https://huggingface.co/DarkKnight1217/RunwayGuard-rtdetr-l/resolve/main/best.pt">resolve/main/best.pt</a></td></tr>
<tr><td><b>sha256</b></td><td><code>C2D2A418069D9AC8658CA85B8359A9E1AB740CE96826C5138178B080F9243269</code></td></tr>
<tr><td><b>Stack</b></td><td>Ultralytics RT-DETR-L 8.4.147 · FastAPI · Docker · imgsz 480</td></tr>
<tr><td><b>Train</b></td><td>Colab Tesla T4 · AdamW <code>lr0=1e-4</code> · batch 8 · seed 42</td></tr>
<tr><td><b>Run</b></td><td>11 epochs logged · ~3.9 h wall time · val mAP50-95 peak ~0.659 at epoch 7 · ship <code>best.pt</code>, not <code>last.pt</code></td></tr>
</table>

---

## 1. Scope

| In scope | Out of scope |
| --- | --- |
| Visible debris candidates in one image (class, box, score) | Runway reopen / Part 139 clearance |
| Structured `/ask` answers grounded in detections | Material chemistry, mass, future engine damage |

FAA AC 150/5210-24 treats detection as one FOD control activity next to prevention, removal, and evaluation. A still image cannot close that loop.

---

## 2. Data decisions

**Source.** FOD-A v2.1 only (MIT, VOC, 33,793 images at 300×300; [arXiv:2110.03072](https://arxiv.org/abs/2110.03072)). No scrapes.

**Rejected publisher split.** ImageSet audit found duplicate list entries, IDs in both trainval and test, and **1,403 identical dHash groups** spanning the official split. That is near-duplicate *risk* (plain asphalt can also collide). We dropped that split rather than risk inflated self-report mAP.

**Replacement.** Grouped split, seed **42** (`scripts/create_grouped_split.py`):

| Split | Images |
| --- | ---: |
| train | 24,103 |
| val | 4,808 |
| test | 4,882 |

Groups use dHash (threshold 18), adjacent IDs, label tuples, and environment tags. Heuristic, not verified shot IDs. ID-disjoint ≠ video-independent. Test stayed frozen after weights and API thresholds were locked.

**Taxonomy.** 31 publisher labels → 7 operational classes in `config/taxonomy.json`:

`fastener_hardware` · `hand_tool` · `flexible_debris` · `loose_metal` · `plastic_paper_debris` · `component_container` · `natural_debris`

The head cannot separate bolt from washer. `/ask` abstains on subtype questions even when the parent class fires.

---

## 3. Grouped-test metrics (frozen)

| Metric | Value | Means | Does not mean |
| --- | ---: | --- | --- |
| mAP50 | **0.755** | Ranking on this split | Hidden RAP score |
| mAP50-95 | **0.636** | Localization under IoU sweep | Calibrated confidence |
| Library P / R | **0.734 / 0.812** | Ultralytics COCO-style means | Exact `/ask` ops at 0.25 / 0.50 |

**Per-class mAP50**

| Class | mAP50 |
| --- | ---: |
| plastic_paper_debris | 0.966 |
| component_container | 0.940 |
| hand_tool | 0.839 |
| fastener_hardware | 0.821 |
| flexible_debris | 0.761 |
| loose_metal | **0.541** |
| natural_debris | **0.418** |

Val rose ~0.03 → ~0.80 mAP50 by epoch 4, then oscillated while train loss kept falling. That curve is diagnostic only. Numbers above are **test**.

---

## 4. Five observed failures

Mined on grouped **test** at conf **0.25**, IoU **0.5** (`scripts/mine_failures.py`).

| ID | Mode | What happened |
| --- | --- | --- |
| `016303` | Small-object miss | Tiny `fastener_hardware` (~0.23% of 300×300). Zero detections. Miss ≠ clear pavement. |
| `022564` | Extra-class FP | True `loose_metal` + high-score `plastic_paper_debris`. `/ask` answers use ≥0.50; evidence still collects at 0.25. |
| `027929` | Class confusion | GT `natural_debris` matched as `loose_metal` (+ plastic). Wrong label; same spot may still need inspect. |
| `027930` | Natural → metal bias | Rock/wood predicted only as `loose_metal`. Matches weak class AP (0.418). |
| `022573` | Loose-metal precision | True metal + high-conf `natural_debris` FP. Fits P/R asymmetry (P 0.492, R 0.968). |

---

## 5. Part B

Intent and arithmetic: `app/reasoning.py` (authoritative). Optional OpenAI HTTP may propose constrained intent JSON only (`app/llm_phrase.py`); the final sentence is always code-rendered. Docker default: `RUNWAYGUARD_LLM=0`. No LangChain / LangGraph / CrewAI / AutoGen.

| Question | Result |
| --- | --- |
| "Is the runway safe to reopen?" | `UNOBSERVABLE` / `INSUFFICIENT_INFORMATION` |
| "Is there a screwdriver?" + `hand_tool`@0.93 | Abstain (class broader than the noun) |

---

## 6. Ready vs not claimed

| Ready | Not claimed |
| --- | --- |
| `/health`, `/detect`, `/ask` | Calibrated probabilities |
| Docker, logging, upload limits, imgsz 480 | API P/R = Ultralytics library means without a separate ops table |
| pytest, HF weights, optional `deploy/` Gradio | Leakage-free video identity |
| | Replacement for authorized inspection |
