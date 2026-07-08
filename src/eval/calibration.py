"""
Expected Calibration Error (ECE) for object detection.

Standard classification ECE bins predictions by confidence and compares
mean confidence to accuracy per bin. For detection, "accuracy" per
prediction is binary: matched to an unclaimed ground-truth box of the same
category at IoU >= --iou-thr (0.5 by default), greedy by descending score —
same matching convention pycocotools/COCOeval uses internally, just
collapsed to one fixed threshold since calibration is a single-threshold
question. Unmatched (background/duplicate) predictions count as incorrect
(0), matching the security-screening framing in PLAN.md RQ4: a
low-confidence prediction that isn't a false alarm should reflect
appropriately low confidence, and a high-confidence prediction should
actually be right most of the time — that IS what operators depend on when
they set a single working threshold.

Usage:
    python src/eval/calibration.py \
        --gt data/processed/pidray_val.json \
        --pred results/predictions/deimv2_val_25ep.json \
        --name DEIMv2 --n-bins 15 --iou-thr 0.5

    # compare multiple models in one table:
    python src/eval/calibration.py \
        --gt data/processed/pidray_val.json \
        --pred results/predictions/deimv2_val_25ep.json:DEIMv2 \
              results/predictions/dfine_val_25ep.json:D-FINE \
              results/predictions/yolo11_val_25ep.json:YOLO11
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def box_iou_xywh(a, b) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match_predictions(gt_by_image_cat: dict, predictions: list, iou_thr: float) -> np.ndarray:
    """Returns a 0/1 correctness array aligned with `predictions` (already
    assumed sorted by descending score within each image+category group by
    the caller)."""
    used = defaultdict(set)  # (image_id, category_id) -> set of matched GT indices
    correct = np.zeros(len(predictions), dtype=np.float32)

    for i, pred in enumerate(predictions):
        key = (pred["image_id"], pred["category_id"])
        gts = gt_by_image_cat.get(key, [])
        best_iou, best_j = 0.0, -1
        for j, gt_box in enumerate(gts):
            if j in used[key]:
                continue
            iou = box_iou_xywh(pred["bbox"], gt_box)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thr:
            used[key].add(best_j)
            correct[i] = 1.0
    return correct


def compute_ece(gt_path: Path, pred_path: Path, iou_thr: float, n_bins: int) -> dict:
    with open(gt_path) as f:
        gt = json.load(f)
    with open(pred_path) as f:
        preds = json.load(f)

    gt_by_image_cat = defaultdict(list)
    for ann in gt["annotations"]:
        gt_by_image_cat[(ann["image_id"], ann["category_id"])].append(ann["bbox"])

    preds_sorted = sorted(preds, key=lambda p: -p["score"])
    correct = match_predictions(gt_by_image_cat, preds_sorted, iou_thr)
    scores = np.array([p["score"] for p in preds_sorted], dtype=np.float32)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_stats = []
    n_total = len(scores)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (scores > lo) & (scores <= hi) if lo > 0 else (scores >= lo) & (scores <= hi)
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = scores[mask].mean()
        ece += (n_in_bin / n_total) * abs(bin_acc - bin_conf)
        bin_stats.append({"range": (float(lo), float(hi)), "n": int(n_in_bin),
                           "accuracy": float(bin_acc), "confidence": float(bin_conf)})

    return {"ece": float(ece), "n_predictions": n_total, "bins": bin_stats}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", nargs="+", required=True,
                         help="either a single path (use --name), or multiple path:Name pairs")
    parser.add_argument("--name", type=str, default="model")
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--n-bins", type=int, default=15)
    args = parser.parse_args()

    results = {}
    for entry in args.pred:
        if ":" in entry:
            path_str, name = entry.rsplit(":", 1)
        else:
            path_str, name = entry, args.name
        results[name] = compute_ece(args.gt, Path(path_str), args.iou_thr, args.n_bins)

    print(f"\n=== ECE @ IoU>={args.iou_thr}, {args.n_bins} bins ===")
    for name, r in results.items():
        print(f"{name:<12} ECE={r['ece']:.4f}  (n={r['n_predictions']} predictions)")


if __name__ == "__main__":
    main()
