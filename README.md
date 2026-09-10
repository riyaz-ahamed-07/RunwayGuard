# RunwayGuard

RunwayGuard is an auditable object-detection and reasoning API for **visible airport foreign-object debris (FOD)**. It fine-tunes RT-DETR on the FOD-A dataset and exposes:

- `POST /detect`: detected class, bounding box, and confidence;
- `POST /ask`: a constrained natural-language layer that routes questions, reasons only over detection output, and abstains when the image cannot support the requested conclusion.

RunwayGuard is an **inspection-prioritization aid**. It does not certify a runway as safe, replace an authorized inspection, infer material or weight, or predict aircraft damage from one image.

## Why this problem

The FAA defines FOD as an object in an inappropriate airport location that can injure personnel or damage aircraft. The problem fits object detection because the observable evidence is the object class and location; operational clearance remains a human decision.

- FAA FOD program: <https://www.faa.gov/airports/airport_safety/fod>
- Official FOD-A repository and MIT-licensed dataset: <https://github.com/FOD-UNOmaha/FOD-data>
- FOD-A paper: <https://arxiv.org/abs/2110.03072>

## Dataset audit

FOD-A v2.1 Pascal VOC contains:

- 33,793 images, all 300 x 300;
- 33,793 parseable XML annotation files;
- 34,472 valid boxes across 31 source labels, mapped to seven operational non-COCO classes;
- no missing images, malformed XML files, or invalid boxes;
- 9,030 small boxes occupying less than 1% of image area.

The supplied split and taxonomy were not used unchanged. Audit findings:

- 19 duplicate lines in `test.txt` and 47 in `trainval.txt`;
- four IDs listed in both files;
- 70 annotated images listed in neither file;
- 1,403 identical, label-compatible perceptual-hash groups crossed the supplied split.

Those findings indicate a substantial risk that closely related video frames inflate evaluation. `scripts/create_grouped_split.py` creates a deterministic split that groups:

1. consecutive frames with matching labels and environmental metadata when their 64-bit difference hashes are similar; and
2. identical perceptual hashes when label and environmental metadata also match.

The output manifests live in `data_splits/`; the original dataset is never modified.

Several source labels had many frames but too few independent capture groups to support leakage-free validation. For example, `Tape`, `Wood`, `Hose`, and `AdjustableWrench` could not all be placed into train, validation, and test without dividing closely related sequences. Fine-grained distinctions such as `Bolt` versus `BoltWasher` also do not change the operational response: inspect and remove the object.

`config/taxonomy.json` therefore maps all 31 source labels exactly once into seven operational classes:

- `fastener_hardware`
- `hand_tool`
- `flexible_debris`
- `loose_metal`
- `plastic_paper_debris`
- `component_container`
- `natural_debris`

All seven final classes have independent validation and test examples. The mapping is explicit and reversible for audit purposes; annotations are not silently discarded.

## Reproduce data preparation

Download **FOD-A v2.1 Pascal VOC (412 MB)** from the official repository, extract it, and identify the directory containing `Annotations`, `JPEGImages`, and `ImageSets`.

```bash
python scripts/create_grouped_split.py \
  --dataset-root /path/to/VOC2007 \
  --output-dir data_splits

python scripts/convert_voc_to_yolo.py \
  --dataset-root /path/to/VOC2007 \
  --splits-dir data_splits \
  --taxonomy config/taxonomy.json \
  --output-dir prepared_data
```

On Windows PowerShell, put each command on one line rather than using the Bash line-continuation characters above.

## Training

Test the complete pipeline for one epoch first:

```bash
python train.py --data prepared_data/data.yaml --smoke-test
```

Then run the recorded baseline:

```bash
python train.py \
  --data prepared_data/data.yaml \
  --model rtdetr-l.pt \
  --epochs 25 \
  --imgsz 480 \
  --batch 8 \
  --device 0 \
  --seed 42
```

The starting checkpoint is COCO-pretrained RT-DETR-L, but the output head is fine-tuned for seven FOD operational classes. These classes are not COCO classes. The training script records Python, PyTorch, CUDA device, Ultralytics version, seed, and all arguments in the run directory.

RT-DETR's CUDA deformable-attention backward operation is not bit-for-bit deterministic. The fixed seed controls initialization, ordering, and augmentation sampling; the project does not claim stronger reproducibility than the implementation provides.

## Evaluation

Do not inspect or tune against the grouped test images until model and confidence decisions are fixed.

```bash
python evaluate.py \
  --weights runs/rtdetr_l_foda/weights/best.pt \
  --data prepared_data/data.yaml \
  --imgsz 480 \
  --batch 8 \
  --device 0
```

Report at least:

- mAP50-95 and mAP50;
- precision and recall;
- per-class AP and the confusion matrix;
- performance by object size, plus wet/dry and bright/dim/dark conditions where feasible;
- five **observed** failure examples with root-cause analysis.

No final metric is claimed in this repository until training and untouched-test evaluation complete.

## Run the API

Place the trained checkpoint at `weights/best.pt`, or set `MODEL_PATH` to its location.

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Interactive documentation: <http://localhost:8000/docs>

### Detection request

```bash
curl -X POST http://localhost:8000/detect \
  -F "image=@sample.jpg" \
  -F "confidence=0.25"
```

Example response shape:

```json
{
  "detections": [
    {
      "class_id": 3,
    "class_name": "fastener_hardware",
      "confidence": 0.9132,
      "box": {"x1": 121.2, "y1": 87.1, "x2": 150.8, "y2": 132.4}
    }
  ],
  "image_size": {"width": 300, "height": 300},
  "model": "best.pt",
  "confidence_threshold": 0.25
}
```

The numbers above illustrate the schema; they are not represented as an actual model result.

### Reasoning request

```bash
curl -X POST http://localhost:8000/ask \
  -F "image=@sample.jpg" \
  -F "question=How many bolts are visible?" \
  -F "confidence=0.25"
```

For `Is the runway safe to reopen?`, the endpoint intentionally returns:

```json
{
  "route": "UNOBSERVABLE",
  "status": "INSUFFICIENT_INFORMATION",
  "answer": "Insufficient information. One image can support detection of visible candidate debris only; it cannot certify runway safety, predict aircraft damage, or determine hidden physical facts. An authorized inspection is still required.",
  "evidence": [],
  "guardrail_reason": "The question asks for information that is not observable from one image."
}
```

## Reasoning policy

The reasoning layer is deliberately hand-written—no LangChain, agent framework, or AutoML system is used.

- Image-grounded count, presence, listing, and most-common-class questions call RT-DETR.
- Unsupported general questions do not call the detector.
- Questions asking for runway certification, future damage, ownership, exact material, weight, or physical distance abstain.
- Detections at or above 0.50 are considered confident for answers. Lower-scored returned candidates make complete counts uncertain.
- A non-detection is never translated into “the runway is safe” or “the object is definitely absent.”

## Known limitations to test

These are hypotheses for targeted testing, not claimed observed failures:

1. tiny FOD occupying less than 1% of the image;
2. wet pavement reflections and painted-line edges causing false positives;
3. low light, glare, blur, or camera compression;
4. visually diverse source objects grouped into the same operational category;
5. novel debris outside the closed 31-class taxonomy;
6. domain shift from staged 300 x 300 data to a different airport, camera height, or lens.

The final two-page memo must replace at least five of these hypotheses with actual examples produced by the trained model.
