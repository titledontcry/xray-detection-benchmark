"""
YOLO11 training entrypoint for the PIDray benchmark — thin wrapper around
ultralytics' Python API, kept for consistency with
src/training/train_{deimv2,dfine}.py (one entrypoint per model, config
loaded from configs/model/<model>/).

Unlike DEIMv2/D-FINE, ultralytics is a pip package (no third_party clone,
no sys.path tricks, no gradient-accumulation need — YOLO11-S's memory
footprint is far smaller than the DETR-style decoders that needed it).

Usage (run from repo root):
    python src/training/train_yolo11.py --cfg configs/model/yolo11/yolo11s_pidray.yaml
"""
import argparse
from pathlib import Path

import albumentations as A
import yaml
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    args, unknown = parser.parse_known_args()

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    # allow ad-hoc overrides, e.g. --epochs 1 for a smoke test
    overrides = {}
    for item in unknown:
        key, _, value = item.lstrip("-").partition("=")
        overrides[key] = yaml.safe_load(value)
    cfg.update(overrides)

    # ultralytics.data.augment.Albumentations reads hyp.augmentations and
    # uses it as its transform list verbatim when given (falls back to its
    # own low-probability Blur/ToGray/CLAHE/... combo otherwise) — a
    # built-in extension point, no monkey-patching needed. Can't express a
    # Python object list in the YAML config, so it's injected here instead
    # of configs/model/yolo11/yolo11s_pidray.yaml. Same clip/grid as
    # src/data/augmentation.py and src/data/clahe_transform.py — one CLAHE
    # definition, three models.
    #
    # clip_limit=(2.0, 2.0), not 2.0: albumentations treats a scalar as a
    # (1, clip_limit) RANGE and samples a random value per call — confirmed
    # via .get_params() returning values scattered across [1, 2]. This
    # preprocessing step must be fixed/deterministic, not randomized.
    cfg["augmentations"] = [A.CLAHE(clip_limit=(2.0, 2.0), tile_grid_size=(8, 8), p=1.0)]

    model_name = cfg.pop("model")
    model = YOLO(model_name)
    model.train(**cfg)


if __name__ == "__main__":
    main()
