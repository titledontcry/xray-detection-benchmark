"""
Validate PIDray's COCO annotations (Phase 1.3).

PIDray ships in COCO format already (confirmed from the official repo's
mmdet config: dataset_type='CocoDataset'), so no conversion is needed —
this script only validates integrity and prints the EDA summary needed
for Phase 1.2 decisions (image sizes -> SAHI needed?, class distribution
-> loss weighting?).

Usage (on server, after scripts/sync_data.sh):
    python src/data/validate_coco.py --data-dir data/raw/pidray
"""
import argparse
import json
from collections import Counter
from pathlib import Path

ANN_FILES = {
    "train": "annotations/xray_train.json",
    "test_easy": "annotations/xray_test_easy.json",
    "test_hard": "annotations/xray_test_hard.json",
    "test_hidden": "annotations/xray_test_hidden.json",
}

IMG_DIRS = {"train": "train", "test_easy": "easy",
            "test_hard": "hard", "test_hidden": "hidden"}


def validate_split(name: str, ann_path: Path, img_dir: Path):
    with open(ann_path) as f:
        coco = json.load(f)

    cats = {c["id"]: c["name"] for c in coco["categories"]}
    n_img, n_ann = len(coco["images"]), len(coco["annotations"])

    # class distribution
    class_counts = Counter(cats[a["category_id"]] for a in coco["annotations"])

    # image size stats (for SAHI decision, RQ2)
    widths = [im["width"] for im in coco["images"]]
    heights = [im["height"] for im in coco["images"]]

    # integrity: every annotation references an existing image id
    img_ids = {im["id"] for im in coco["images"]}
    orphans = sum(1 for a in coco["annotations"] if a["image_id"] not in img_ids)

    # integrity: image files exist on disk (sample-check first 200 to stay fast)
    missing = 0
    for im in coco["images"][:200]:
        if not (img_dir / im["file_name"]).exists():
            missing += 1

    print(f"\n=== {name} ===")
    print(f"images: {n_img} | annotations: {n_ann} | categories: {len(cats)}")
    print(f"image width  min/max: {min(widths)}/{max(widths)}")
    print(f"image height min/max: {min(heights)}/{max(heights)}")
    print(f"orphan annotations: {orphans} (must be 0)")
    print(f"missing files in first 200 sampled: {missing} (must be 0)")
    print("class distribution:")
    for cls, cnt in class_counts.most_common():
        print(f"  {cls:<12} {cnt}")
    return cats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    all_cats = None
    for name, rel in ANN_FILES.items():
        ann_path = args.data_dir / rel
        if not ann_path.exists():
            print(f"[WARN] missing: {ann_path} — check extraction")
            continue
        cats = validate_split(name, ann_path, args.data_dir / IMG_DIRS[name])
        if all_cats is None:
            all_cats = cats
        else:
            assert cats == all_cats, f"category mismatch in {name}!"

    print("\nAll splits share identical category mapping." if all_cats else "")


if __name__ == "__main__":
    main()
