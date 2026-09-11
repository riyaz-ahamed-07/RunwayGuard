# RunwayGuard — start training now

Local work is ready. Your laptop’s PyTorch is **CPU-only**, so full RT-DETR training must run on **Google Colab (GPU)** or Kaggle.

## Upload this zip

`C:\Users\thahs\Documents\Codex\2026-09-11\create-an-image-of\RunwayGuard_project.zip`

## Colab steps (in order)

1. Open [Google Colab](https://colab.research.google.com/) → **Runtime → Change runtime type → GPU (T4)**.
2. Upload `RunwayGuard_project.zip` (or put it on Drive and copy the path).
3. Upload / open `RunwayGuard/RunwayGuard_Training.ipynb` and run cells **top to bottom**.
4. Cell order:
   - Install deps + unzip project
   - Download FOD-A VOC zip via `gdown` (file id `1RdErcq8PGRXZUOGauaACkQG44T-QyZ4x`)
   - Convert with **our** `data_splits/` + `config/taxonomy.json` (do not use the official Roboflow/VOC split)
   - **Smoke:** 1 epoch on 2% of data
   - Mount Drive → **full train** (see time budget below)
   - Evaluate on the **grouped test** split only after training finishes
5. Keep the Colab tab awake. Checkpoint path:

`/content/drive/MyDrive/RunwayGuardRuns/rtdetr_l_foda/weights/best.pt`

### Time budget (T4)

Full 25 epochs is often **8–12 hours**. If you only have **~2–3 hours**, interrupt the long run and start a shorter baseline (document this in the memo):

```bash
!cd /content/RunwayGuard && python train.py \
  --data /content/foda_yolo/data.yaml \
  --model rtdetr-l.pt \
  --epochs 8 \
  --imgsz 416 \
  --batch 12 \
  --device 0 \
  --seed 42 \
  --patience 3 \
  --project /content/drive/MyDrive/RunwayGuardRuns \
  --name rtdetr_l_foda_fast
```

If CUDA OOM, use `--batch 8` (or `4`) and keep `imgsz 416`. Prefer a finished 8-epoch model over an unfinished 25-epoch run.

## After training (do not invent results)

1. Copy `best.pt` + `results.csv` + `test_metrics.json` into the local project.
2. Inspect real false positives/negatives; pick **five** failures with filenames.
3. Wire weights into the FastAPI (`weights/best.pt`), smoke-test `/detect` and `/ask`.
4. Write the 2-page memo from *your* metrics and failures.
5. Upload weights (GitHub Release / Drive / HF) and submit to SharePoint.

## Decisions already locked (yours to defend)

| Decision | Choice |
|---|---|
| Problem | Airport runway FOD screening (not “runway is safe”) |
| Dataset | FOD-A v2.1 only for training |
| Split | Grouped perceptual-hash split (official split leaks) |
| Classes | 7 operational taxonomy mapped from 31 source labels |
| Model | Ultralytics RT-DETR-L fine-tune |
| Guardrail | Abstain on safety certification; low-conf → insufficient information |
