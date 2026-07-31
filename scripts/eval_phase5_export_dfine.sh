#!/usr/bin/env bash
# Run ON THE SERVER, inside the `dfine` conda env, from repo root.
# Phase 5: export D-FINE predictions (all 3 seeds) on the official PIDray
# test split (easy/hard/hidden) to plain COCO-format result JSONs, consumed
# by src/eval/compute_metrics.py, tide_eval.py, calibration.py,
# bootstrap_significance.py. Never touches the official test JSONs
# themselves (read-only) — CLAUDE.md hard rule #1.
#
# Prefers best_stg2.pth (D-FINE/DEIM's 2-stage schedule: stg2 is the later,
# more-refined fine-tuning phase, so its best checkpoint supersedes
# best_stg1.pth) but falls back to best_stg1.pth if a given seed never wrote
# a best_stg2.pth (that file is only written when stage 2 actually beats
# stage 1's best — confirmed to happen for DEIMv2 seed1, 2026-08-01, so the
# same defensive check is applied here even though all 3 D-FINE seeds
# happened to have best_stg2.pth). Also skips any (seed, split) whose output
# file already exists, so re-running after a partial failure doesn't redo
# finished work.
#
# Committed as a script, not pasted, per the 2026-07-28 paste-corruption
# lesson (see PLAN.md Decision Log).
set -euo pipefail

for seed in 0 1 2; do
  ckpt_dir="outputs/dfine_hgnetv2_s_pidray_phase4/seed${seed}"
  if [ -f "${ckpt_dir}/best_stg2.pth" ]; then
    ckpt="${ckpt_dir}/best_stg2.pth"
  else
    ckpt="${ckpt_dir}/best_stg1.pth"
  fi
  for split in easy hard hidden; do
    out="results/predictions/phase5/dfine_${split}_seed${seed}.json"
    if [ -f "${out}" ]; then
      echo "=== D-FINE seed${seed} / ${split} — already done, skipping ==="
      continue
    fi
    echo "=== D-FINE seed${seed} / ${split} (using ${ckpt}) ==="
    python src/eval/export_deim_predictions.py \
      --repo-root third_party/D-FINE \
      -c configs/model/dfine/dfine_pidray.yml \
      -r "${ckpt}" \
      --ann-file "data/processed/pidray_test_${split}.json" \
      --img-dir "data/raw/pidray/${split}" \
      --out "${out}"
  done
done
