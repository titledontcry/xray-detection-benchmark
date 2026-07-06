"""
X-ray-specific preprocessing: CLAHE contrast enhancement (+ optional SAHI
tiling for wide images — likely unnecessary for PIDray's per-item images,
revisit once real image dimensions are inspected).

Usage:
    python src/data/preprocess.py --in-dir data/raw/pidray/images \
        --out-dir data/processed/images --clahe-clip 2.0 --clahe-grid 8
"""
import argparse
from pathlib import Path

import cv2


def apply_clahe(img_gray, clip_limit=2.0, tile_grid_size=8):
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                             tileGridSize=(tile_grid_size, tile_grid_size))
    return clahe.apply(img_gray)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--clahe-clip", type=float, default=2.0)
    parser.add_argument("--clahe-grid", type=int, default=8)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    images = list(args.in_dir.glob("*.jpg")) + list(args.in_dir.glob("*.png"))
    print(f"Found {len(images)} images to preprocess.")

    for img_path in images:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        enhanced = apply_clahe(img, args.clahe_clip, args.clahe_grid)
        out_path = args.out_dir / img_path.name
        cv2.imwrite(str(out_path), enhanced)

    print(f"Wrote preprocessed images to {args.out_dir}")


if __name__ == "__main__":
    main()
