"""
Phase 1.6 (still-open part): convert the materialized COCO split JSONs
(src/data/materialize_split.py output — already 0-indexed category ids,
shared with DEIMv2/D-FINE) into YOLO-format labels for YOLO11/ultralytics.

Ultralytics resolves an image's label file by substring-replacing
"/images/" with "/labels/" in its path (img2label_paths()). Our raw images
live in a flat data/raw/pidray/train/ folder with no "images" segment, and
we don't want to write into data/raw/ (read-only official data) or copy
~29k images into a new folder (wasteful). Instead this script SYMLINKS each
image into data/processed/yolo/images/{split}/ so ultralytics' path
substitution lands on the real label files at
data/processed/yolo/labels/{split}/ — no image bytes duplicated.

Usage:
    python src/data/convert_coco_to_yolo.py \
        --processed-dir data/processed \
        --raw-image-dir data/raw/pidray/train \
        --out-dir data/processed/yolo
"""
import argparse
import json
from pathlib import Path


def convert_split(coco_json_path: Path, raw_image_dir: Path, out_dir: Path, split: str):
    with open(coco_json_path) as f:
        coco = json.load(f)

    images_dir = out_dir / "images" / split
    labels_dir = out_dir / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    anns_by_image = {}
    for a in coco["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)

    n_boxes = 0
    for im in coco["images"]:
        src = raw_image_dir / im["file_name"]
        link = images_dir / im["file_name"]
        if not link.exists():
            link.symlink_to(src.resolve())

        w, h = im["width"], im["height"]
        lines = []
        for a in anns_by_image.get(im["id"], []):
            x, y, bw, bh = a["bbox"]
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            nw, nh = bw / w, bh / h
            lines.append(f"{a['category_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        n_boxes += len(lines)

        label_path = labels_dir / (Path(im["file_name"]).stem + ".txt")
        label_path.write_text("\n".join(lines))

    print(f"{split}: {len(coco['images'])} images | {n_boxes} boxes -> {images_dir}, {labels_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, required=True,
                         help="dir containing pidray_train.json / pidray_val.json from materialize_split.py")
    parser.add_argument("--raw-image-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    for split in ("train", "val"):
        convert_split(args.processed_dir / f"pidray_{split}.json",
                      args.raw_image_dir, args.out_dir, split)


if __name__ == "__main__":
    main()
