#!/usr/bin/env bash
# Run ON THE SERVER. Downloads PIDray from the official public Google Drive link.
# Data NEVER goes through git.
#
# PEP 668-safe: modern Ubuntu blocks `pip install` into system Python,
# so this script uses a dedicated lightweight venv (.venv-tools) for
# download utilities only. Model training envs are managed separately
# by conda (scripts/setup_env.sh).
set -euo pipefail

DEST="data/raw/pidray"
VENV=".venv-tools"
GDRIVE_ID="1UMq0CP20lKcraOTvsFMjiLjPfDam9jAp"  # from official PIDray README

mkdir -p "$DEST"

# --- 1. Ensure tools venv exists ---
if [ ! -d "$VENV" ]; then
  echo "Creating tools venv at $VENV ..."
  if ! python3 -m venv "$VENV" 2>/dev/null; then
    echo "ERROR: python3-venv not available. Install it first:"
    echo "  sudo apt install python3-venv python3-full"
    exit 1
  fi
fi
"$VENV/bin/pip" install --quiet --upgrade gdown

# --- 2. Download (skip if already present) ---
ZIP="$DEST/pidray.zip"
if [ -f "$ZIP" ]; then
  echo "$ZIP already exists — skipping download."
else
  "$VENV/bin/gdown" "$GDRIVE_ID" -O "$ZIP"
fi

# --- 3. Extract (skip if already extracted) ---
if [ -d "$DEST/annotations" ]; then
  echo "Already extracted — skipping."
else
  echo "Extracting..."
  unzip -q "$ZIP" -d "$DEST"
fi

echo ""
echo "Expected structure:"
echo "  $DEST/annotations/xray_train.json + xray_test_{easy,hard,hidden}.json"
echo "  $DEST/{train,easy,hard,hidden}/  (image folders)"
echo ""
echo "Next step:"
echo "  python src/data/validate_coco.py --data-dir $DEST"