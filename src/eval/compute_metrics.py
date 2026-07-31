"""
Phase 5: full COCOeval metric suite (AP@0.5:0.95, AP@0.5, AP@0.75, AP-small,
AP-medium, AP-large, AR@100) for one or more exported prediction sets against
a ground-truth split — the same table format used for the Phase 2/Phase 4
val-set comparisons (there, the numbers came straight out of each model's own
training log; here there is no training log, since this is inference-only
export against the official test JSONs, so this script fills that gap using
the same pycocotools COCOeval already proven correct in
bootstrap_significance.py's coco_ap()).

Usage — single model:
    python src/eval/compute_metrics.py \
        --gt data/processed/pidray_test_easy.json \
        --pred results/predictions/phase5/dfine_easy_seed0.json --name D-FINE_seed0

Usage — many at once (prints one table, optionally dumps JSON for aggregation):
    python src/eval/compute_metrics.py \
        --gt data/processed/pidray_test_easy.json \
        --pred results/predictions/phase5/dfine_easy_seed0.json:D-FINE_seed0 \
              results/predictions/phase5/dfine_easy_seed1.json:D-FINE_seed1 \
        --out-json results/metrics/phase5_easy.json
"""
import argparse
import contextlib
import io
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

STAT_NAMES = ["AP@.5:.95", "AP@.5", "AP@.75", "AP-S", "AP-M", "AP-L",
              "AR@1", "AR@10", "AR@100", "AR-S", "AR-M", "AR-L"]


def compute_stats(gt_path: Path, pred_path: Path) -> list:
    with open(pred_path) as f:
        preds = json.load(f)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(gt_path))
        if len(preds) == 0:
            return [0.0] * len(STAT_NAMES)
        coco_dt = coco_gt.loadRes(preds)
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return [float(s) for s in ev.stats]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", nargs="+", required=True,
                         help="either a single bare path (use --name), or multiple path:Name pairs")
    parser.add_argument("--name", type=str, default="model")
    parser.add_argument("--out-json", type=Path, default=None,
                         help="optional: dump {name: {stat_name: value}} for later mean/std aggregation")
    args = parser.parse_args()

    results = {}
    for entry in args.pred:
        if ":" in entry:
            path_str, name = entry.rsplit(":", 1)
        else:
            path_str, name = entry, args.name
        stats = compute_stats(args.gt, Path(path_str))
        results[name] = dict(zip(STAT_NAMES, stats))

    header = f"{'name':<20}" + "".join(f"{s:>10}" for s in STAT_NAMES[:6])
    print(header)
    for name, r in results.items():
        row = f"{name:<20}" + "".join(f"{r[s]:>10.4f}" for s in STAT_NAMES[:6])
        print(row)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[compute_metrics] wrote {args.out_json}")


if __name__ == "__main__":
    main()
