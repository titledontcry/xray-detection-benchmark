"""
Phase 1.4: split PIDray's OFFICIAL TRAIN set into train/val (85/15).

Official test (easy/hard/hidden) is kept untouched for literature
comparability — never re-split, never used before Phase 5.

Multi-label stratified because one image can contain several of the 12
categories and the distribution is long-tailed.

Usage (on server):
    pip install iterative-stratification
    python src/data/stratified_split.py \
        --train-ann data/raw/pidray/annotations/xray_train.json \
        --out-dir data/splits --val-frac 0.15 --seed 42

Outputs (small JSON id lists -> COMMIT THESE TO GIT):
    data/splits/train_ids.json
    data/splits/val_ids.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-ann", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.train_ann) as f:
        coco = json.load(f)

    cat_ids = sorted(c["id"] for c in coco["categories"])
    cat_index = {cid: i for i, cid in enumerate(cat_ids)}

    image_ids = [im["id"] for im in coco["images"]]
    id_index = {iid: i for i, iid in enumerate(image_ids)}

    # multi-hot label matrix: images x categories
    Y = np.zeros((len(image_ids), len(cat_ids)), dtype=np.int8)
    for ann in coco["annotations"]:
        Y[id_index[ann["image_id"]], cat_index[ann["category_id"]]] = 1

    X = np.arange(len(image_ids)).reshape(-1, 1)
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=args.val_frac, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(X, Y))

    train_ids = [image_ids[i] for i in train_idx]
    val_ids = [image_ids[i] for i in val_idx]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "train_ids.json", "w") as f:
        json.dump(sorted(train_ids), f)
    with open(args.out_dir / "val_ids.json", "w") as f:
        json.dump(sorted(val_ids), f)

    # per-class sanity report
    print(f"train: {len(train_ids)} images | val: {len(val_ids)} images")
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    print(f"{'class':<12} {'train':>7} {'val':>6} {'val%':>6}")
    for cid in cat_ids:
        col = cat_index[cid]
        tr = int(Y[train_idx, col].sum())
        va = int(Y[val_idx, col].sum())
        pct = 100 * va / max(tr + va, 1)
        print(f"{cat_names[cid]:<12} {tr:>7} {va:>6} {pct:>5.1f}%")
    print(f"\nSeed={args.seed}, val_frac={args.val_frac}. "
          f"Commit data/splits/*.json to git.")


if __name__ == "__main__":
    main()
