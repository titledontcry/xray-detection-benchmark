"""
Convert raw PIDray annotations into the project's COCO JSON master format.

This is the single source of truth that DEIMv2, D-FINE, and YOLO11 (after a
further COCO->YOLO conversion step) all read from — keeping one master format
avoids subtle per-model annotation drift.

Usage:
    python src/data/convert_to_coco.py \
        --raw-dir data/raw/pidray \
        --out data/processed/pidray_coco.json
"""
import argparse
import json
from pathlib import Path

PIDRAY_CATEGORIES = [
    "gun", "knife", "wrench", "pliers", "scissors", "hammer",
    "handcuffs", "baton", "sprayer", "powerbank", "lighter", "bullet",
]


def build_categories():
    return [{"id": i + 1, "name": name, "supercategory": "prohibited_item"}
             for i, name in enumerate(PIDRAY_CATEGORIES)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    # TODO: replace with actual PIDray annotation parsing once repo structure
    # is confirmed (some PIDray releases already ship near-COCO JSON — verify
    # before writing a from-scratch parser).
    coco = {
        "images": [],
        "annotations": [],
        "categories": build_categories(),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(coco, f)
    print(f"Wrote COCO JSON skeleton to {args.out}")


if __name__ == "__main__":
    main()
