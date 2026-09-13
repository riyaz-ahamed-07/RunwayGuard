"""Operating-point metrics and error-mining helpers (synthetic-testable)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    confidence: float = 1.0


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def match_detections(
    predictions: list[Box],
    ground_truth: list[Box],
    *,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.25,
) -> dict[str, object]:
    """Greedy one-to-one matching for operating-point P/R and error kinds."""
    preds = sorted(
        [p for p in predictions if p.confidence >= score_threshold],
        key=lambda item: item.confidence,
        reverse=True,
    )
    remaining = set(range(len(ground_truth)))
    true_positive = 0
    false_positive = 0
    class_confusion = 0
    localization_error = 0
    records: list[dict[str, object]] = []

    for pred in preds:
        best_idx = None
        best_iou = 0.0
        for gt_idx in remaining:
            gt_candidate = ground_truth[gt_idx]
            if gt_candidate.class_id != pred.class_id:
                continue
            score = iou(pred, gt_candidate)
            if score > best_iou:
                best_iou = score
                best_idx = gt_idx
        if best_idx is None or best_iou < iou_threshold:
            # Prefer labeling same-location wrong-class overlaps as class_confusion.
            confusion_idx = None
            confusion_iou = 0.0
            for gt_idx in remaining:
                score = iou(pred, ground_truth[gt_idx])
                if score > confusion_iou:
                    confusion_iou = score
                    confusion_idx = gt_idx
            if confusion_idx is not None and confusion_iou >= iou_threshold:
                class_confusion += 1
                false_positive += 1
                records.append(
                    {
                        "kind": "class_confusion",
                        "confidence": pred.confidence,
                        "iou": confusion_iou,
                    }
                )
            else:
                false_positive += 1
                records.append({"kind": "false_positive", "confidence": pred.confidence, "iou": best_iou})
            continue
        remaining.remove(best_idx)
        if best_iou < 0.75:
            localization_error += 1
            true_positive += 1
            records.append({"kind": "localization_weak_tp", "confidence": pred.confidence, "iou": best_iou})
        else:
            true_positive += 1
            records.append({"kind": "true_positive", "confidence": pred.confidence, "iou": best_iou})

    false_negative = len(remaining)
    for gt_idx in remaining:
        records.append({"kind": "false_negative", "class_id": ground_truth[gt_idx].class_id})

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else None
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else None
    return {
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "class_confusion": class_confusion,
        "localization_weak_tp": localization_error,
        "precision": precision,
        "recall": recall,
        "records": records,
    }


def operating_point_table(
    predictions: list[Box],
    ground_truth: list[Box],
    thresholds: tuple[float, ...] = (0.25, 0.50),
) -> list[dict[str, object]]:
    return [match_detections(predictions, ground_truth, score_threshold=threshold) for threshold in thresholds]
