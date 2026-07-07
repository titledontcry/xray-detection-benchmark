"""
Phase 5 prep: apply the same 0-indexed category remap as
materialize_split.py to the official PIDray test sets (easy/hard/hidden).

NEVER writes to the original raw files — outputs a separate copy under
data/processed/. data/raw/pidray/annotations/xray_test_*.json is the
untouchable official test split (CLAUDE.md hard rule #1: never train,
tune, or modify it before Phase 5 evaluation).

The remap must be built from the SAME source (the official train
annotation file's categories list) as materialize_split.py, so a label
means the same thing whether a model sees train, val, or test data — see
materialize_split.py's docstring for why the remap exists at all
(DEIMv2/D-FINE's CocoDetection uses raw category_id as class label
whenever remap_mscoco_category=False, and PIDray's ids run 1-12, not
0-indexed).

Usage:
    python src/data/remap_test_categories.py \
        --train-ann data/raw/pidray/annotations/xray_train.json \
        --raw-dir data/raw/pidray/annotations \
        --out-dir data/processed
"""
import argparse
import json
from pathlib import Path

from materialize_split import materialize

TEST_SPLITS = {
    "test_easy": "xray_test_easy.json",
    "test_hard": "xray_test_hard.json",
    "test_hidden": "xray_test_hidden.json",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-ann", type=Path, required=True,
                         help="official train ann file — source of the category id->label mapping")
    parser.add_argument("--raw-dir", type=Path, required=True,
                         help="dir containing xray_test_{easy,hard,hidden}.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    with open(args.train_ann) as f:
        train_coco = json.load(f)
    cat_id_to_label = {c["id"]: i for i, c in enumerate(train_coco["categories"])}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, filename in TEST_SPLITS.items():
        with open(args.raw_dir / filename) as f:
            coco = json.load(f)

        all_ids = {im["id"] for im in coco["images"]}
        remapped = materialize(coco, all_ids, cat_id_to_label)

        out_path = args.out_dir / f"pidray_{split}.json"
        with open(out_path, "w") as f:
            json.dump(remapped, f)

        print(f"{split}: {len(remapped['images'])} images | "
              f"{len(remapped['annotations'])} annotations -> {out_path}")


if __name__ == "__main__":
    main()
