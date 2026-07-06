#!/usr/bin/env bash
# Run this ON THE SERVER, once, after `git clone` / `git pull`.
set -euo pipefail

echo "Creating 3 isolated conda environments (DEIMv2, D-FINE, YOLO11)..."
conda env create -f environments/deimv2.yml
conda env create -f environments/dfine.yml
conda env create -f environments/yolo11.yml

echo "Cloning model repos into third_party/ (not tracked by our git, gitignored)..."
mkdir -p third_party
[ -d third_party/DEIMv2 ] || git clone https://github.com/Intellindust-AI-Lab/DEIMv2.git third_party/DEIMv2
[ -d third_party/D-FINE ] || git clone https://github.com/Peterande/D-FINE.git third_party/D-FINE

echo "Done. Activate with: conda activate deimv2 | dfine | yolo11"
