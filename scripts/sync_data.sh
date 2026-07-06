#!/usr/bin/env bash
# Run ON THE SERVER. Downloads PIDray from the official public Google Drive link.
# Data NEVER goes through git.
set -euo pipefail

DEST="data/raw/pidray"
mkdir -p "$DEST"

# Official download (from https://github.com/bywang2018/security-dataset README):
#   Google Drive file id: 1UMq0CP20lKcraOTvsFMjiLjPfDam9jAp
pip install -q gdown
gdown --id 1UMq0CP20lKcraOTvsFMjiLjPfDam9jAp -O "$DEST/pidray.zip"

echo "Extracting..."
unzip -q "$DEST/pidray.zip" -d "$DEST"

echo "Expected structure:"
echo "  $DEST/annotations/xray_train.json + xray_test_{easy,hard,hidden}.json"
echo "  $DEST/{train,easy,hard,hidden}/  (image folders)"
echo "Next: python src/data/validate_coco.py --data-dir $DEST"
