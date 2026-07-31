#!/usr/bin/env bash
# Run ON THE SERVER, inside the `deimv2` conda env, from repo root.
# Phase 5: export DEIMv2 predictions (all 3 seeds) on the official PIDray
# test split (easy/hard/hidden) to plain COCO-format result JSONs, consumed
# by src/eval/compute_metrics.py, tide_eval.py, calibration.py,
# bootstrap_significance.py. Never touches the official test JSONs
# themselves (read-only) — CLAUDE.md hard rule #1.
#
# Prefers best_stg2.pth (same 2-stage-schedule rationale as the D-FINE export
# script) but falls back to best_stg1.pth when stg2 doesn't exist — the
# stgN-best checkpoint is only written when that stage actually beats the
# prior stage's best, so a missing best_stg2.pth means stage 1's checkpoint
# is the true best for that seed (confirmed: seed1 has no best_stg2.pth,
# 2026-08-01). Also skips any (seed, split) whose output file already
# exists, so re-running after a partial failure doesn't redo finished work.
#
# Committed as a script, not pasted, per the 2026-07-28 paste-corruption
# lesson (see PLAN.md Decision Log).
set -euo pipefail

for seed in 0 1 2; do
  ckpt_dir="outputs/deimv2_hgnetv2_s_pidray_phase4/seed${seed}"
  if [ -f "${ckpt_dir}/best_stg2.pth" ]; then
    ckpt="${ckpt_dir}/best_stg2.pth"
  else
    ckpt="${ckpt_dir}/best_stg1.pth"
  fi
  for split in easy hard hidden; do
    out="results/predictions/phase5/deimv2_${split}_seed${seed}.json"
    if [ -f "${out}" ]; then
      echo "=== DEIMv2 seed${seed} / ${split} — already done, skipping ==="
      continue
    fi
    echo "=== DEIMv2 seed${seed} / ${split} (using ${ckpt}) ==="
    python src/eval/export_deim_predictions.py \
      --repo-root third_party/DEIMv2 \
      -c configs/model/deimv2/deimv2_pidray.yml \
      -r "${ckpt}" \
      --ann-file "data/processed/pidray_test_${split}.json" \
      --img-dir "data/raw/pidray/${split}" \
      --out "${out}"
  done
done
