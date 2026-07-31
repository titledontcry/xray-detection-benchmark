#!/usr/bin/env bash
# Run ON THE SERVER, in any conda env that has pycocotools (e.g. `dfine` —
# already a training dependency, since each model's own log.txt AP numbers
# come from pycocotools internally), from repo root.
#
# Phase 5: full COCO metric suite (AP@0.5:0.95, AP@0.5, AP@0.75, AP-S/M/L)
# for all 3 models x 3 seeds, per official test split (easy/hard/hidden).
# Requires all 27 prediction files from scripts/eval_phase5_export_*.sh to
# already exist. Writes results/metrics/phase5_{split}.json for later
# mean/std aggregation across seeds.
set -euo pipefail

mkdir -p results/metrics

for split in easy hard hidden; do
  echo "=== metrics: ${split} ==="
  python src/eval/compute_metrics.py \
    --gt "data/processed/pidray_test_${split}.json" \
    --pred \
      "results/predictions/phase5/dfine_${split}_seed0.json:D-FINE_seed0" \
      "results/predictions/phase5/dfine_${split}_seed1.json:D-FINE_seed1" \
      "results/predictions/phase5/dfine_${split}_seed2.json:D-FINE_seed2" \
      "results/predictions/phase5/deimv2_${split}_seed0.json:DEIMv2_seed0" \
      "results/predictions/phase5/deimv2_${split}_seed1.json:DEIMv2_seed1" \
      "results/predictions/phase5/deimv2_${split}_seed2.json:DEIMv2_seed2" \
      "results/predictions/phase5/yolo11_${split}_seed0.json:YOLO11_seed0" \
      "results/predictions/phase5/yolo11_${split}_seed1.json:YOLO11_seed1" \
      "results/predictions/phase5/yolo11_${split}_seed2.json:YOLO11_seed2" \
    --out-json "results/metrics/phase5_${split}.json"
done
