#!/usr/bin/env bash
# Run ON THE SERVER, inside the `yolo11` conda env, from repo root.
# Phase 4 final training: YOLO11-S, 3 seeds, full ultralytics-default 100
# epochs, best hyperparameters from Phase 3 HPO (see
# configs/model/yolo11/phase4_best_hpo.yaml).
#
# Written as a committed script (not a pasted multi-line command) because the
# 2026-07-22 Phase 4 launch attempt invoked train_yolo11.py directly without
# the `python` prefix ("Permission denied" — file isn't executable) and the
# surrounding for-loop got corrupted in the terminal paste ("syntax error near
# unexpected token `done`"), so no seed ever started.
set -euo pipefail

for seed in 0 1 2; do
  echo "=== YOLO11-S seed ${seed} ==="
  python src/training/train_yolo11.py \
    --cfg configs/model/yolo11/yolo11s_pidray.yaml \
    --seed="${seed}" \
    --name="yolo11s_pidray_phase4/seed${seed}" \
    --lr0=0.00268210629989773 \
    --weight_decay=4.4559451018018605e-05 \
    --box=9.624363569814477 \
    --cls=0.6552574942507758 \
    --mosaic=0.7798713039813766
done
