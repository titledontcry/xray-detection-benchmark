#!/usr/bin/env bash
# Run ON THE SERVER, inside the `deimv2` conda env, from repo root.
# Phase 5: export DEIMv2 predictions (all 3 seeds) on the official PIDray
# test split (easy/hard/hidden) to plain COCO-format result JSONs, consumed
# by src/eval/compute_metrics.py, tide_eval.py, calibration.py,
# bootstrap_significance.py. Never touches the official test JSONs
# themselves (read-only) — CLAUDE.md hard rule #1.
#
# Uses best_stg2.pth (same 2-stage-schedule rationale as the D-FINE export
# script). Committed as a script, not pasted, per the 2026-07-28
# paste-corruption lesson (see PLAN.md Decision Log).
set -euo pipefail

for seed in 0 1 2; do
  for split in easy hard hidden; do
    echo "=== DEIMv2 seed${seed} / ${split} ==="
    python src/eval/export_deim_predictions.py \
      --repo-root third_party/DEIMv2 \
      -c configs/model/deimv2/deimv2_pidray.yml \
      -r "outputs/deimv2_hgnetv2_s_pidray_phase4/seed${seed}/best_stg2.pth" \
      --ann-file "data/processed/pidray_test_${split}.json" \
      --img-dir "data/raw/pidray/${split}" \
      --out "results/predictions/phase5/deimv2_${split}_seed${seed}.json"
  done
done
