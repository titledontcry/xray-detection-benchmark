#!/usr/bin/env bash
# Run ON THE SERVER, in a conda env with pycocotools (e.g. `dfine`), from
# repo root. Usage: bash scripts/eval_phase5_bootstrap.sh [n_bootstrap]
#
# Phase 5: paired bootstrap significance test for all 3 model pairs, per
# split, using the same representative-seed choice as eval_phase5_tide.sh
# (diagnostic tool, not the headline metric — see PLAN.md Decision Log
# 2026-08-01).
#
# WARNING: bootstrap_significance.py re-runs a full COCOeval per resample
# per model (its own docstring calls n_bootstrap=200 "a quick check", but
# D-FINE/DEIMv2's prediction files here are unusually large — low --conf
# threshold at export time means far more boxes per image than typical —
# so 200 resamples may take much longer than that docstring assumes).
# RUN A SMOKE TEST FIRST with a small n (e.g. `bash scripts/eval_phase5_bootstrap.sh 10`)
# to see real per-comparison timing before committing to a bigger n on all
# 9 (3 splits x 3 pairs) comparisons.
set -euo pipefail

N="${1:-200}"

declare -A DFINE_SEED=( [easy]=0 [hard]=0 [hidden]=0 )
declare -A DEIMV2_SEED=( [easy]=0 [hard]=1 [hidden]=2 )
declare -A YOLO11_SEED=( [easy]=2 [hard]=1 [hidden]=2 )

mkdir -p results/bootstrap

for split in easy hard hidden; do
  gt="data/processed/pidray_test_${split}.json"
  dfine="results/predictions/phase5/dfine_${split}_seed${DFINE_SEED[$split]}.json"
  deimv2="results/predictions/phase5/deimv2_${split}_seed${DEIMV2_SEED[$split]}.json"
  yolo11="results/predictions/phase5/yolo11_${split}_seed${YOLO11_SEED[$split]}.json"

  echo "=== Bootstrap ${split}: DEIMv2 vs D-FINE (n=${N}) ==="
  python src/eval/bootstrap_significance.py --gt "$gt" \
    --pred-a "$deimv2" --name-a DEIMv2 \
    --pred-b "$dfine" --name-b D-FINE \
    --n-bootstrap "$N" | tee "results/bootstrap/${split}_deimv2_vs_dfine.txt"

  echo "=== Bootstrap ${split}: DEIMv2 vs YOLO11 (n=${N}) ==="
  python src/eval/bootstrap_significance.py --gt "$gt" \
    --pred-a "$deimv2" --name-a DEIMv2 \
    --pred-b "$yolo11" --name-b YOLO11 \
    --n-bootstrap "$N" | tee "results/bootstrap/${split}_deimv2_vs_yolo11.txt"

  echo "=== Bootstrap ${split}: D-FINE vs YOLO11 (n=${N}) ==="
  python src/eval/bootstrap_significance.py --gt "$gt" \
    --pred-a "$dfine" --name-a D-FINE \
    --pred-b "$yolo11" --name-b YOLO11 \
    --n-bootstrap "$N" | tee "results/bootstrap/${split}_dfine_vs_yolo11.txt"
done
