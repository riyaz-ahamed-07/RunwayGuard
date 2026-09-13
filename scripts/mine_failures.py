"""Mine real grouped-test failure examples for the RAP memo.

Prioritizes weak classes and small GT boxes, then prints concrete image IDs.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ultralytics import RTDETR

from app.eval_ops import Box, match_detections

CLASS_NAMES = {
    0: "fastener_hardware",
    1: "hand_tool",
    2: "flexible_debris",
    3: "loose_metal",
    4: "plastic_paper_debris",
    5: "component_container",
    6: "natural_debris",
}


def parse_label(path: Path, img_w: float = 300.0, img_h: float = 300.0) -> list[Box]:
    boxes: list[Box] = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:])
        w, h = bw * img_w, bh * img_h
        x1 = xc * img_w - w / 2
        y1 = yc * img_h - h / 2
        boxes.append(Box(x1=x1, y1=y1, x2=x1 + w, y2=y1 + h, class_id=cls, confidence=1.0))
    return boxes


def gt_area_frac(box: Box, img_w: float = 300.0, img_h: float = 300.0) -> float:
    return max(0.0, box.x2 - box.x1) * max(0.0, box.y2 - box.y1) / (img_w * img_h)


def score_priority(gts: list[Box]) -> float:
    if not gts:
        return 0.0
    weak = sum(1 for g in gts if g.class_id in {3, 6})
    tiny = sum(1 for g in gts if gt_area_frac(g) < 0.01)
    return weak * 10 + tiny * 3 + len(gts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path("weights/best.pt"))
    parser.add_argument("--data-root", type=Path, default=Path("prepared_data"))
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=350)
    parser.add_argument("--output", type=Path, default=Path("docs/artifacts/failure_mine.json"))
    args = parser.parse_args()

    images_dir = args.data_root / "images" / "test"
    labels_dir = args.data_root / "labels" / "test"
    candidates: list[tuple[float, Path]] = []
    for img_path in sorted(images_dir.glob("*.jpg")):
        gts = parse_label(labels_dir / f"{img_path.stem}.txt")
        candidates.append((score_priority(gts), img_path))
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = [path for _, path in candidates[: args.limit]]

    print(f"selected {len(selected)} prioritized test images", flush=True)
    print(f"loading {args.weights}", flush=True)
    model = RTDETR(str(args.weights))
    buckets: dict[str, list[dict]] = defaultdict(list)

    for idx, img_path in enumerate(selected, start=1):
        print(f"infer {idx}/{len(selected)} {img_path.stem}", flush=True)
        gts = parse_label(labels_dir / f"{img_path.stem}.txt")
        result = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )[0]
        preds: list[Box] = []
        if result.boxes is not None and len(result.boxes):
            xyxy = result.boxes.xyxy.cpu().tolist()
            cls = result.boxes.cls.cpu().tolist()
            conf = result.boxes.conf.cpu().tolist()
            for box, c, s in zip(xyxy, cls, conf):
                preds.append(
                    Box(
                        x1=float(box[0]),
                        y1=float(box[1]),
                        x2=float(box[2]),
                        y2=float(box[3]),
                        class_id=int(c),
                        confidence=float(s),
                    )
                )

        matched = match_detections(preds, gts, iou_threshold=0.5, score_threshold=args.conf)
        image_id = img_path.stem
        tiny_gt = [g for g in gts if gt_area_frac(g) < 0.01]
        gt_names = [CLASS_NAMES[g.class_id] for g in gts]
        pred_names = [CLASS_NAMES[p.class_id] for p in preds]

        if matched["false_negative"] and tiny_gt:
            buckets["small_object_miss"].append(
                {
                    "id": image_id,
                    "gt": gt_names,
                    "pred": pred_names,
                    "fn": matched["false_negative"],
                    "tiny_gt": len(tiny_gt),
                }
            )
        if matched["false_positive"] and not gts:
            buckets["background_fp"].append({"id": image_id, "pred": pred_names})
        if matched["false_positive"] and gts:
            buckets["false_positive"].append(
                {
                    "id": image_id,
                    "gt": gt_names,
                    "pred": pred_names,
                    "fp": matched["false_positive"],
                    "max_pred_conf": max((p.confidence for p in preds), default=0.0),
                }
            )
        if matched["class_confusion"]:
            buckets["class_confusion"].append(
                {
                    "id": image_id,
                    "gt": gt_names,
                    "pred": pred_names,
                    "confusions": matched["class_confusion"],
                }
            )
        if any(g.class_id == 6 for g in gts) and matched["false_negative"]:
            buckets["natural_debris_miss"].append(
                {"id": image_id, "gt": gt_names, "pred": pred_names, "fn": matched["false_negative"]}
            )
        if any(g.class_id == 3 for g in gts) and matched["false_positive"]:
            buckets["loose_metal_fp_context"].append(
                {"id": image_id, "gt": gt_names, "pred": pred_names, "fp": matched["false_positive"]}
            )
        if matched["false_negative"] and not tiny_gt:
            buckets["miss_non_tiny"].append(
                {"id": image_id, "gt": gt_names, "pred": pred_names, "fn": matched["false_negative"]}
            )

        if idx % 25 == 0:
            print(f"processed {idx}/{len(selected)}")

    summary = {kind: items[:12] for kind, items in buckets.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: len(v) for k, v in buckets.items()}, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
