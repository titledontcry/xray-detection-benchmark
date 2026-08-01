#!/usr/bin/env bash
# Run ON THE SERVER, in a conda env with `tidecv` installed
# (`pip install tidecv` once if missing), from repo root.
#
# Phase 5: TIDE error-type breakdown (Cls/Loc/Both/Dupe/Bkg/Miss) per split,
# one representative seed per model — not all 3, since TIDE is a diagnostic/
# interpretive tool, not the headline AP metric (see PLAN.md Decision Log
# 2026-08-01). Representative seed = whichever seed's AP@0.5:0.95 is closest
# to that model's 3-seed mean on that split.
set -euo pipefail

declare -A DFINE_SEED=( [easy]=0 [hard]=0 [hidden]=0 )
declare -A DEIMV2_SEED=( [easy]=0 [hard]=1 [hidden]=2 )
declare -A YOLO11_SEED=( [easy]=2 [hard]=1 [hidden]=2 )

for split in easy hard hidden; do
  echo "=== TIDE: ${split} ==="
  python src/eval/tide_eval.py \
    --gt "data/processed/pidray_test_${split}.json" \
    --pred \
      "results/predictions/phase5/dfine_${split}_seed${DFINE_SEED[$split]}.json:D-FINE" \
      "results/predictions/phase5/deimv2_${split}_seed${DEIMV2_SEED[$split]}.json:DEIMv2" \
      "results/predictions/phase5/yolo11_${split}_seed${YOLO11_SEED[$split]}.json:YOLO11" \
    --out-dir "results/tide/phase5_${split}" \
    | tee "results/tide/phase5_${split}_summary.txt"
done
