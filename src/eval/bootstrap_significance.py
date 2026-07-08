"""
Paired bootstrap significance test for AP@0.5:0.95 differences between two
models — answers "is model A's higher AP just noise from which images
happen to be in the val/test set, or a real difference?"

Method: resample images WITH REPLACEMENT (same size as the original split),
recompute AP@0.5:0.95 for both models on that resampled set via a fresh
pycocotools COCOeval each iteration, record the paired difference
(AP_A - AP_B). Repeat --n-bootstrap times. Because both models are scored
on the *same* resampled image set each iteration (paired), per-image
difficulty variance cancels out — this is a stricter test than comparing
independent confidence intervals for A and B separately.

Report: mean/std of the bootstrap difference distribution, and a two-sided
bootstrap p-value (fraction of resamples where the sign of the difference
flips relative to the observed difference — i.e. how often "the other
model would have won").

This is slow by construction (COCOeval re-run N times) — default
n_bootstrap=200 for a quick check; use 1000+ for the number that actually
goes in the paper.

Usage:
    python src/eval/bootstrap_significance.py \
        --gt data/processed/pidray_val.json \
        --pred-a results/predictions/deimv2_val_25ep.json --name-a DEIMv2 \
        --pred-b results/predictions/yolo11_val_25ep.json --name-b YOLO11 \
        --n-bootstrap 200
"""
import argparse
import contextlib
import copy
import io
import json
import random
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def resample_coco(gt: dict, preds_a: list, preds_b: list, image_ids: list):
    """Builds resampled GT + prediction dicts where each drawn image_id gets
    a fresh unique pseudo-id (so repeats from sampling-with-replacement
    don't collide in COCOeval, which requires unique image ids)."""
    images_by_id = {im["id"]: im for im in gt["images"]}
    anns_by_image = {}
    for ann in gt["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    preds_a_by_image, preds_b_by_image = {}, {}
    for p in preds_a:
        preds_a_by_image.setdefault(p["image_id"], []).append(p)
    for p in preds_b:
        preds_b_by_image.setdefault(p["image_id"], []).append(p)

    new_images, new_anns_a_gt, new_preds_a, new_preds_b = [], [], [], []
    next_img_id, next_ann_id = 1, 1
    for orig_id in image_ids:
        pseudo_id = next_img_id
        next_img_id += 1
        im = dict(images_by_id[orig_id])
        im["id"] = pseudo_id
        new_images.append(im)
        for ann in anns_by_image.get(orig_id, []):
            a = dict(ann)
            a["image_id"] = pseudo_id
            a["id"] = next_ann_id
            next_ann_id += 1
            new_anns_a_gt.append(a)
        for p in preds_a_by_image.get(orig_id, []):
            pp = dict(p)
            pp["image_id"] = pseudo_id
            new_preds_a.append(pp)
        for p in preds_b_by_image.get(orig_id, []):
            pp = dict(p)
            pp["image_id"] = pseudo_id
            new_preds_b.append(pp)

    gt_resampled = {"images": new_images, "annotations": new_anns_a_gt,
                     "categories": gt["categories"]}
    return gt_resampled, new_preds_a, new_preds_b


def coco_ap(gt_dict: dict, preds: list, quiet: bool = True) -> float:
    # pycocotools prints "creating index... index created!" etc. on every
    # call with no built-in quiet flag — silenced here since --n-bootstrap
    # resamples call this hundreds of times and the per-iteration AP table
    # isn't useful, only the final aggregate stats printed by main() are.
    ctx = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
    with ctx:
        coco_gt = COCO()
        coco_gt.dataset = gt_dict
        coco_gt.createIndex()
        if len(preds) == 0:
            return 0.0
        coco_dt = coco_gt.loadRes(preds)
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev.stats[0]  # AP@0.5:0.95


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred-a", type=Path, required=True)
    parser.add_argument("--pred-b", type=Path, required=True)
    parser.add_argument("--name-a", type=str, default="A")
    parser.add_argument("--name-b", type=str, default="B")
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.gt) as f:
        gt = json.load(f)
    with open(args.pred_a) as f:
        preds_a = json.load(f)
    with open(args.pred_b) as f:
        preds_b = json.load(f)

    all_image_ids = [im["id"] for im in gt["images"]]
    n = len(all_image_ids)

    observed_gt, observed_a, observed_b = resample_coco(gt, preds_a, preds_b, all_image_ids)
    ap_a_observed = coco_ap(observed_gt, observed_a)
    ap_b_observed = coco_ap(observed_gt, observed_b)
    observed_diff = ap_a_observed - ap_b_observed
    print(f"\nObserved: {args.name_a} AP={ap_a_observed:.4f}  {args.name_b} AP={ap_b_observed:.4f}  "
          f"diff={observed_diff:+.4f}")

    rng = random.Random(args.seed)
    diffs = []
    for i in range(args.n_bootstrap):
        sample_ids = [rng.choice(all_image_ids) for _ in range(n)]
        gt_r, preds_a_r, preds_b_r = resample_coco(gt, preds_a, preds_b, sample_ids)
        ap_a = coco_ap(gt_r, preds_a_r)
        ap_b = coco_ap(gt_r, preds_b_r)
        diffs.append(ap_a - ap_b)
        if (i + 1) % 20 == 0:
            print(f"[bootstrap] {i + 1}/{args.n_bootstrap} done", flush=True)

    mean_diff = sum(diffs) / len(diffs)
    std_diff = (sum((d - mean_diff) ** 2 for d in diffs) / len(diffs)) ** 0.5
    # two-sided bootstrap p-value: how often the resampled diff crosses zero
    # relative to the observed diff's sign
    if observed_diff >= 0:
        p_value = sum(1 for d in diffs if d <= 0) / len(diffs)
    else:
        p_value = sum(1 for d in diffs if d >= 0) / len(diffs)
    p_value = min(1.0, 2 * p_value)  # two-sided

    print(f"\n=== Bootstrap result ({args.n_bootstrap} resamples) ===")
    print(f"mean diff = {mean_diff:+.4f}, std = {std_diff:.4f}")
    print(f"two-sided p-value ~= {p_value:.4f} "
          f"({'significant at p<0.05' if p_value < 0.05 else 'NOT significant at p<0.05'})")


if __name__ == "__main__":
    main()
