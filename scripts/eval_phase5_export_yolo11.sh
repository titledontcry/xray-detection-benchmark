#!/usr/bin/env bash
# Run ON THE SERVER, inside the `yolo11` conda env, from repo root.
# Phase 5: export YOLO11-S predictions (all 3 seeds) on the official PIDray
# test split (easy/hard/hidden) to plain COCO-format result JSONs, consumed
# by src/eval/compute_metrics.py, tide_eval.py, calibration.py,
# bootstrap_significance.py. Never touches the official test JSONs
# themselves (read-only) — CLAUDE.md hard rule #1.
#
# Skips any (seed, split) whose output file already exists, so re-running
# after a partial failure doesn't redo finished work. Committed as a script,
# not pasted, per the 2026-07-28 paste-corruption lesson (see PLAN.md
# Decision Log).
set -euo pipefail

for seed in 0 1 2; do
  for split in easy hard hidden; do
    out="results/predictions/phase5/yolo11_${split}_seed${seed}.json"
    if [ -f "${out}" ]; then
      echo "=== YOLO11-S seed${seed} / ${split} — already done, skipping ==="
      continue
    fi
    echo "=== YOLO11-S seed${seed} / ${split} ==="
    python src/eval/export_yolo11_predictions.py \
      --weights "runs/detect/outputs/yolo11s_pidray_phase4/seed${seed}/weights/best.pt" \
      --ann-file "data/processed/pidray_test_${split}.json" \
      --img-dir "data/raw/pidray/${split}" \
      --out "${out}"
  done
done
