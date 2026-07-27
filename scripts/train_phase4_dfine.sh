#!/usr/bin/env bash
# Run ON THE SERVER, inside the `dfine` conda env, from repo root.
# Phase 4 final training: D-FINE, 3 seeds, full paper-default 132 epochs,
# best hyperparameters from Phase 3 HPO (see configs/model/dfine/phase4_best_hpo.yaml).
#
# Written as a committed script (not a pasted multi-line command) because the
# 2026-07-22 Phase 4 launch attempt corrupted the pasted -u override string in
# the terminal (two dotted keys got mashed into a single bogus
# "warmuDFINECriterion" key, crashing every seed before training started).
set -euo pipefail

for seed in 0 1 2; do
  echo "=== D-FINE seed ${seed} ==="
  torchrun --nproc_per_node=1 src/training/train_dfine.py \
    -c configs/model/dfine/dfine_pidray.yml \
    --seed "${seed}" \
    --output-dir "outputs/dfine_hgnetv2_s_pidray_phase4/seed${seed}" \
    -u optimizer.lr=0.00034213513273286 \
       optimizer.weight_decay=0.000253790402093864 \
       lr_warmup_scheduler.warmup_duration=250 \
       DFINECriterion.weight_dict.loss_fgl=0.11374730115262209
done
