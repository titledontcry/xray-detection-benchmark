"""
Shared on-the-fly preprocessing + augmentation policy (Phase 1.5).

This is the single source of truth for how every model (DEIMv2, D-FINE,
YOLO11) must preprocess/augment PIDray images. Each model's training repo
(cloned into third_party/ in Phase 2) must wrap its own dataloader to call
into this module rather than relying on its own repo's default transforms —
that is the only way the 3-way comparison stays valid (CLAUDE.md hard rule:
"Identical augmentation + preprocessing across all 3 models").

Decision (2026-07-07): CLAHE runs on-the-fly (not pre-computed to disk),
so HPO can sweep clip/grid without re-processing ~47k images per trial,
and disk usage stays flat. See PLAN.md Decision Log.

Split between two stages:
  - `clahe_only`: deterministic contrast normalization, applied identically
    to train/val/test — this is preprocessing, not augmentation.
  - `train_augmentations`: random flip + bbox-safe crop, TRAIN SPLIT ONLY.
    Deliberately excludes color jitter / HSV jitter (X-ray domain rule —
    color has no physical meaning here, so a model must not learn to rely
    on it). Each model repo's own defaults must be overridden to match
    this (e.g. YOLO11/ultralytics hyp: hsv_h=0.0, hsv_s=0.0, hsv_v=0.0).

Usage:
    from src.data.augmentation import build_train_pipeline, build_eval_pipeline

    train_tf = build_train_pipeline(clahe_clip=2.0, clahe_grid=8)
    eval_tf = build_eval_pipeline(clahe_clip=2.0, clahe_grid=8)

    out = train_tf(image=img, bboxes=coco_bboxes, category_ids=cat_ids)
    img, bboxes, cat_ids = out["image"], out["bboxes"], out["category_ids"]
"""
import albumentations as A


def _clahe_only(clip_limit: float, tile_grid_size: int) -> list:
    return [A.CLAHE(clip_limit=clip_limit,
                     tile_grid_size=(tile_grid_size, tile_grid_size),
                     p=1.0)]


def build_train_pipeline(clahe_clip: float = 2.0, clahe_grid: int = 8,
                          flip_prob: float = 0.5,
                          crop_prob: float = 0.5) -> A.Compose:
    """CLAHE + flip/crop only. NO color jitter / NO HSV jitter — X-ray rule."""
    transforms = _clahe_only(clahe_clip, clahe_grid) + [
        A.HorizontalFlip(p=flip_prob),
        A.BBoxSafeRandomCrop(erosion_rate=0.0, p=crop_prob),
    ]
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"],
                                  min_visibility=0.1),
    )


def build_eval_pipeline(clahe_clip: float = 2.0,
                         clahe_grid: int = 8) -> A.Compose:
    """CLAHE only — val/test must never see flip/crop/color augmentation."""
    return A.Compose(
        _clahe_only(clahe_clip, clahe_grid),
        bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"]),
    )
