"""
Phase 2 prep: turn the id-list split (data/splits/*.json, Phase 1.4) into
real per-split COCO JSON files that model training repos can point
`ann_file` at directly.

Why this exists: data/splits/train_ids.json and val_ids.json only store
image ids (small, git-tracked, human-auditable). But every model repo's
dataloader (DEIMv2/D-FINE's CocoDetection, and the COCO->YOLO converter for
YOLO11) expects one COCO JSON per split, not an id list layered on top of
the original file. This script is the one place that materializes that —
every model reads its train/val data from the SAME two output files here,
which keeps the 3-way comparison valid.

The official test split (easy/hard/hidden) is NEVER touched by this script —
those stay as their original raw files straight from data/raw/pidray/.

Usage:
    python src/data/materialize_split.py \
        --train-ann data/raw/pidray/annotations/xray_train.json \
        --split-dir data/splits \
        --out-dir data/processed
"""
import argparse
import json
from pathlib import Path


def materialize(coco: dict, ids: set) -> dict:
    images = [im for im in coco["images"] if im["id"] in ids]
    annotations = [a for a in coco["annotations"] if a["image_id"] in ids]
    return {
        "images": images,
        "annotations": annotations,
        "categories": coco["categories"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-ann", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    with open(args.train_ann) as f:
        coco = json.load(f)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        with open(args.split_dir / f"{split}_ids.json") as f:
            ids = set(json.load(f))

        subset = materialize(coco, ids)
        out_path = args.out_dir / f"pidray_{split}.json"
        with open(out_path, "w") as f:
            json.dump(subset, f)

        print(f"{split}: {len(subset['images'])} images | "
              f"{len(subset['annotations'])} annotations -> {out_path}")


if __name__ == "__main__":
    main()
