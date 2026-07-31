#!/usr/bin/env bash
# Run ON THE SERVER, inside the `dfine` conda env, from repo root.
# Phase 5: export D-FINE predictions (all 3 seeds) on the official PIDray
# test split (easy/hard/hidden) to plain COCO-format result JSONs, consumed
# by src/eval/compute_metrics.py, tide_eval.py, calibration.py,
# bootstrap_significance.py. Never touches the official test JSONs
# themselves (read-only) — CLAUDE.md hard rule #1.
#
# Uses best_stg2.pth (D-FINE/DEIM's 2-stage schedule: stg2 is the later,
# more-refined fine-tuning phase, so its best checkpoint supersedes
# best_stg1.pth) — committed as a script, not pasted, per the 2026-07-28
# paste-corruption lesson (see PLAN.md Decision Log).
set -euo pipefail

for seed in 0 1 2; do
  for split in easy hard hidden; do
    echo "=== D-FINE seed${seed} / ${split} ==="
    python src/eval/export_deim_predictions.py \
      --repo-root third_party/D-FINE \
      -c configs/model/dfine/dfine_pidray.yml \
      -r "outputs/dfine_hgnetv2_s_pidray_phase4/seed${seed}/best_stg2.pth" \
      --ann-file "data/processed/pidray_test_${split}.json" \
      --img-dir "data/raw/pidray/${split}" \
      --out "results/predictions/phase5/dfine_${split}_seed${seed}.json"
  done
done
