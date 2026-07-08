"""
TIDE (Toolbox for Identifying Detection Errors, dbolya/tide) error-type
breakdown — the object-detection-appropriate substitute for a confusion
matrix decided against in PLAN.md (a confusion matrix needs a fixed
confidence/IoU threshold to define TP/FP/FN, which makes it non-objective
and inconsistent across models with different score calibration; TIDE
decomposes AP loss into classification / localization / duplicate /
background / missed-GT error without needing a threshold).

Consumes the plain COCO-format prediction lists produced by
src/eval/export_yolo11_predictions.py and export_deim_predictions.py, run
against the ground truth split (data/processed/pidray_val.json during this
Phase 2 dry run — data/processed/pidray_test_{easy,hard,hidden}.json only in
the real Phase 5 run, never before).

Install (once, in whichever conda env this is run from):
    pip install tidecv

Usage — single model:
    python src/eval/tide_eval.py \
        --gt data/processed/pidray_val.json \
        --pred results/predictions/deimv2_val_25ep.json \
        --name DEIMv2 --out-dir results/tide/deimv2_val_25ep

Usage — compare all 3 at once (prints one combined table):
    python src/eval/tide_eval.py \
        --gt data/processed/pidray_val.json \
        --pred results/predictions/deimv2_val_25ep.json:DEIMv2 \
              results/predictions/dfine_val_25ep.json:D-FINE \
              results/predictions/yolo11_val_25ep.json:YOLO11 \
        --out-dir results/tide/25ep_comparison
"""
import argparse
from pathlib import Path

from tidecv import TIDE, datasets


def run_one(gt_path: Path, pred_path: Path, name: str, out_dir: Path) -> dict:
    tide = TIDE()
    gt = datasets.COCO(str(gt_path))
    pred = datasets.COCOResult(str(pred_path))
    tide.evaluate_range(gt, pred, name=name, mode=TIDE.BOX)
    tide.summarize()
    out_dir.mkdir(parents=True, exist_ok=True)
    tide.plot(str(out_dir))
    # tide.errors[name] holds the per-error-type dAP breakdown after evaluate
    run = tide.runs[name]
    return {err_type: run.error_dAPs.get(err_type, 0.0) for err_type in
            ["Cls", "Loc", "Both", "Dupe", "Bkg", "Miss"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", nargs="+", required=True,
                         help="either a single path (use --name), or multiple path:Name pairs to compare")
    parser.add_argument("--name", type=str, default="model",
                         help="only used when --pred is a single bare path")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    results = {}
    for entry in args.pred:
        if ":" in entry:
            path_str, name = entry.rsplit(":", 1)
        else:
            path_str, name = entry, args.name
        results[name] = run_one(args.gt, Path(path_str), name, args.out_dir / name)

    if len(results) > 1:
        print("\n=== TIDE error-type comparison (dAP, lower = less of that error) ===")
        header = ["Model", "Cls", "Loc", "Both", "Dupe", "Bkg", "Miss"]
        print(f"{header[0]:<12}" + "".join(f"{h:>8}" for h in header[1:]))
        for name, errs in results.items():
            print(f"{name:<12}" + "".join(f"{errs[k]:>8.2f}" for k in header[1:]))

    print(f"\nPer-model plots saved under {args.out_dir}/<model_name>/")


if __name__ == "__main__":
    main()
