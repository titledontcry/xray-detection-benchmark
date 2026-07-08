"""
Exports DEIMv2 or D-FINE predictions on a PIDray split (val, or later
official test) to a plain COCO-format detection results list — same output
format as src/eval/export_yolo11_predictions.py, so tide_eval.py,
calibration.py, and bootstrap_significance.py work unmodified across all 3
models.

Shared by both repos since D-FINE is the ancestor DEIMv2 forked from (nearly
identical `YAMLConfig` / `cfg.model.deploy()` / `cfg.postprocessor.deploy()`
API) — pass --repo-root to pick which one. Adapted from each repo's own
third_party/{DEIMv2,D-FINE}/tools/inference/torch_inf.py (demo script for a
single image), generalized to loop over an entire COCO split and, critically,
inserting the same CLAHE preprocessing step
(clip=2.0, grid=8x8, src/data/augmentation.py) that val_dataloader used
during training — torch_inf.py's own preprocessing (Resize + ToTensor only)
does NOT include this, since it's a generic single-image demo unaware of our
project's locked preprocessing decision. Skipping it here would silently
evaluate the model on a different input distribution than it was
trained/validated on.

NAMING COLLISION WARNING (same as src/training/train_dfine.py): D-FINE's own
package is named `src`. This script does not import anything from our own
src/ package after sys.path.insert, so it is safe as written — do not add a
bare `from src.data... import ...` here without switching to importlib
file-based loading first.

Usage (run from repo root, matching conda env active):
    python src/eval/export_deim_predictions.py \
        --repo-root third_party/DEIMv2 \
        -c configs/model/deimv2/deimv2_pidray.yml \
        -r outputs/deimv2_hgnetv2_s_pidray_25ep_advisor_checkin/best_stg1.pth \
        --ann-file data/processed/pidray_val.json \
        --img-dir data/raw/pidray/train \
        --out results/predictions/deimv2_val_25ep.json

    python src/eval/export_deim_predictions.py \
        --repo-root third_party/D-FINE \
        -c configs/model/dfine/dfine_pidray.yml \
        -r outputs/dfine_hgnetv2_s_pidray_25ep_advisor_checkin/best_stg1.pth \
        --ann-file data/processed/pidray_val.json \
        --img-dir data/raw/pidray/train \
        --out results/predictions/dfine_val_25ep.json
"""
import argparse
import json
import sys
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True,
                         help="third_party/DEIMv2 or third_party/D-FINE")
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, required=True,
                         help="checkpoint .pth (best_stg1.pth recommended over last.pth)")
    parser.add_argument("--ann-file", type=Path, required=True)
    parser.add_argument("--img-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("-d", "--device", type=str, default="cuda")
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo_root))
    if (args.repo_root / "engine").exists():
        from engine.core import YAMLConfig  # DEIMv2
    else:
        from src.core import YAMLConfig  # D-FINE — see naming collision warning above

    cfg = YAMLConfig(args.config, resume=args.resume)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    checkpoint = torch.load(args.resume, map_location="cpu")
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    device = args.device
    model = Model().to(device).eval()
    img_size = cfg.yaml_cfg["eval_spatial_size"]
    vit_backbone = cfg.yaml_cfg.get("DINOv3STAs", False)  # locked decision uses HGNetV2-S, so False

    clahe = A.CLAHE(clip_limit=(2.0, 2.0), tile_grid_size=(8, 8), p=1.0)
    resize = T.Resize(img_size)
    to_tensor = T.ToTensor()
    normalize = (T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                 if vit_backbone else T.Lambda(lambda x: x))

    with open(args.ann_file) as f:
        coco = json.load(f)
    images_meta = coco["images"]

    predictions = []
    with torch.no_grad():
        for i, im in enumerate(images_meta):
            path = args.img_dir / im["file_name"]
            im_pil = Image.open(path).convert("RGB")
            w, h = im_pil.size

            arr = clahe(image=np.array(im_pil))["image"]
            im_pil = Image.fromarray(arr)
            im_data = normalize(to_tensor(resize(im_pil))).unsqueeze(0).to(device)
            orig_size = torch.tensor([[w, h]]).to(device)

            labels, boxes, scores = model(im_data, orig_size)
            labels, boxes, scores = labels[0], boxes[0], scores[0]
            for lbl, box, score in zip(labels.tolist(), boxes.tolist(), scores.tolist()):
                x1, y1, x2, y2 = box
                predictions.append({
                    "image_id": im["id"],
                    "category_id": int(lbl),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": score,
                })

            if (i + 1) % 200 == 0:
                print(f"[export] {i + 1}/{len(images_meta)} images done", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(predictions, f)
    print(f"[export] wrote {len(predictions)} predictions to {args.out}")


if __name__ == "__main__":
    main()
