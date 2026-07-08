"""
Exports YOLO11 predictions on a PIDray split (val, or later official test) to
a plain COCO-format detection results list — the common input format that
src/eval/tide_eval.py, calibration.py, and bootstrap_significance.py all
consume, so the same downstream scripts work unmodified across all 3 models.

Does NOT use ultralytics' built-in `save_json=True` on model.val(): that
path derives each prediction's "image_id" from the image filename's stem
(int(Path(filename).stem) if numeric else the stem string) — PIDray
filenames are not guaranteed to match our materialized COCO JSON's integer
`image["id"]` field, and a silent mismatch there would corrupt every
downstream metric without raising an error. Instead this script reads
image_id -> file_name directly from the same materialized JSON
(data/processed/pidray_val.json) used to train/evaluate the other two
models, so all 3 exporters key predictions identically.

Usage (run from repo root, yolo11 conda env):
    python src/eval/export_yolo11_predictions.py \
        --weights outputs/yolo11s_pidray_25ep_advisor_checkin/weights/best.pt \
        --ann-file data/processed/pidray_val.json \
        --img-dir data/raw/pidray/train \
        --out results/predictions/yolo11_val_25ep.json
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--ann-file", type=Path, required=True,
                         help="materialized COCO JSON (0-indexed categories), e.g. data/processed/pidray_val.json")
    parser.add_argument("--img-dir", type=Path, required=True,
                         help="folder containing the actual image files (data/raw/pidray/train)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.001,
                         help="low threshold — COCOeval/TIDE need the full score distribution, not just confident boxes")
    args = parser.parse_args()

    with open(args.ann_file) as f:
        coco = json.load(f)
    images = coco["images"]  # each has "id" and "file_name"

    model = YOLO(str(args.weights))

    predictions = []
    batch_size = 32
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        paths = [str(args.img_dir / im["file_name"]) for im in batch]
        results = model.predict(paths, conf=args.conf, verbose=False)
        for im, result in zip(batch, results):
            boxes = result.boxes
            for xyxy, score, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
                x1, y1, x2, y2 = xyxy
                predictions.append({
                    "image_id": im["id"],
                    "category_id": int(cls),  # already 0-indexed, matches materialize_split.py's remap
                    "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO format: [x, y, w, h]
                    "score": score,
                })
        print(f"[export] {min(start + batch_size, len(images))}/{len(images)} images done", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(predictions, f)
    print(f"[export] wrote {len(predictions)} predictions to {args.out}")


if __name__ == "__main__":
    main()
