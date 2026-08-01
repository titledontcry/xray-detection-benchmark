#!/usr/bin/env bash
# Run ON THE SERVER, in any conda env with numpy (e.g. `dfine`), from repo
# root.
#
# Phase 5: Expected Calibration Error (ECE) per split, one representative
# seed per model — same choice as eval_phase5_tide.sh/eval_phase5_bootstrap.sh
# (diagnostic tool, not the headline metric — see PLAN.md Decision Log
# 2026-08-01). Directly relevant to RQ4 (security-screening framing):
# operators rely on confidence scores being trustworthy at a single working
# threshold.
set -euo pipefail

mkdir -p results/calibration

declare -A DFINE_SEED=( [easy]=0 [hard]=0 [hidden]=0 )
declare -A DEIMV2_SEED=( [easy]=0 [hard]=1 [hidden]=2 )
declare -A YOLO11_SEED=( [easy]=2 [hard]=1 [hidden]=2 )

for split in easy hard hidden; do
  echo "=== ECE: ${split} ==="
  python src/eval/calibration.py \
    --gt "data/processed/pidray_test_${split}.json" \
    --pred \
      "results/predictions/phase5/dfine_${split}_seed${DFINE_SEED[$split]}.json:D-FINE" \
      "results/predictions/phase5/deimv2_${split}_seed${DEIMV2_SEED[$split]}.json:DEIMv2" \
      "results/predictions/phase5/yolo11_${split}_seed${YOLO11_SEED[$split]}.json:YOLO11" \
    --iou-thr 0.5 --n-bins 15 \
    | tee "results/calibration/phase5_${split}_summary.txt"
done
