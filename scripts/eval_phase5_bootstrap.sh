#!/usr/bin/env bash
# Run ON THE SERVER, in a conda env with pycocotools (e.g. `dfine`), from
# repo root. Usage: bash scripts/eval_phase5_bootstrap.sh [n_bootstrap] [splits]
#   splits = space-separated list, default "easy hard hidden", e.g. "hidden"
#
# Phase 5: paired bootstrap significance test for all 3 model pairs, per
# split, using the same representative-seed choice as eval_phase5_tide.sh
# (diagnostic tool, not the headline metric — see PLAN.md Decision Log
# 2026-08-01).
#
# A 2026-08-01 n=10 smoke test (all 3 easy-split pairs) measured ~6-7 min per
# comparison — the docstring's own "n=200 is a quick check" assumption does
# NOT hold here (D-FINE/DEIMv2's prediction files are unusually large — low
# --conf export threshold means far more boxes per image than typical), so
# n=200 across all 9 (3 splits x 3 pairs) comparisons would take ~20 hours.
# Scope narrowed to Hidden only (`bash scripts/eval_phase5_bootstrap.sh 100 hidden`)
# since that's the only split with a ranking surprise (D-FINE < YOLO11) that
# actually needs a significance test — Easy/Hard rankings already match
# every seed with small std, nothing there is in question.
set -euo pipefail

N="${1:-200}"
SPLITS="${2:-easy hard hidden}"

declare -A DFINE_SEED=( [easy]=0 [hard]=0 [hidden]=0 )
declare -A DEIMV2_SEED=( [easy]=0 [hard]=1 [hidden]=2 )
declare -A YOLO11_SEED=( [easy]=2 [hard]=1 [hidden]=2 )

mkdir -p results/bootstrap

for split in ${SPLITS}; do
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
