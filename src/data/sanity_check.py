"""
Phase 1.6 sanity check: draw bboxes on N random images and save to disk
for eyeball verification (annotation alignment, no class swaps).

Usage (on server):
    python src/data/sanity_check.py \
        --ann data/raw/pidray/annotations/xray_train.json \
        --img-dir data/raw/pidray/train \
        --out-dir outputs/sanity --n 100 --seed 42
"""
import argparse
import json
import random
from pathlib import Path

import cv2

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
          (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
          (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", type=Path, required=True)
    parser.add_argument("--img-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.ann) as f:
        coco = json.load(f)

    cats = {c["id"]: (i, c["name"]) for i, c in enumerate(coco["categories"])}
    anns_by_img = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    random.seed(args.seed)
    # prefer images that actually have annotations
    candidates = [im for im in coco["images"] if im["id"] in anns_by_img]
    sample = random.sample(candidates, min(args.n, len(candidates)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for im in sample:
        img = cv2.imread(str(args.img_dir / im["file_name"]))
        if img is None:
            print(f"[WARN] unreadable: {im['file_name']}")
            continue
        for a in anns_by_img[im["id"]]:
            x, y, w, h = map(int, a["bbox"])  # COCO bbox = [x, y, w, h]
            ci, cname = cats[a["category_id"]]
            color = COLORS[ci % len(COLORS)]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, cname, (x, max(y - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imwrite(str(args.out_dir / im["file_name"]), img)

    print(f"Wrote {len(sample)} annotated images to {args.out_dir} — "
          f"inspect them manually before proceeding to Phase 2.")


if __name__ == "__main__":
    main()
