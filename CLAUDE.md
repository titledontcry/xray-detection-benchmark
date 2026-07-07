# CLAUDE.md — Project context for Claude Code

## What this project is
Publication-quality benchmark of 3 object detectors on X-ray security imagery:
**DEIMv2, D-FINE** (DETR-based) vs **YOLO11** (CNN one-stage), trained/evaluated on
**PIDray** (public baggage X-ray dataset, 12 classes, COCO format).

Full plan, phase checklist, and decision log live in **PLAN.md** — read it before
making non-trivial changes. Update its checkboxes and Decision Log when tasks
complete or decisions change.

## Current state (update as project progresses)
- Phase 1 (data prep) in progress on the GPU server
- Data downloaded to `data/raw/pidray/` (22GB); had nested `pidray/pidray/` zip
  structure + `__MACOSX` junk — already cleaned
- Next: run `src/data/validate_coco.py`, then `src/data/stratified_split.py`

## Hard rules — never violate
1. **Official PIDray test split (easy/hard/hidden) is untouchable.** Never train,
   tune, or early-stop on it. It is only used in Phase 5 evaluation.
2. **Only the official train set gets split**: train 85% / val 15%,
   multi-label stratified, **seed=42**. Split id lists live in `data/splits/*.json`
   and ARE committed to git.
3. **No color jitter / no HSV augmentation** — X-ray imagery only allows
   flip/crop. Must actively disable these in each model repo's defaults
   (YOLO11/ultralytics enables HSV by default).
4. **Identical augmentation + preprocessing across all 3 models** — otherwise the
   comparison is invalid. Single source of truth: `configs/data/pidray.yaml`.
5. **Raw data and checkpoints never go into git** (see .gitignore). Only code,
   configs, and small split-manifest JSONs are committed.
6. **One COCO JSON is the master annotation format.** YOLO txt files are always
   regenerated from it, never hand-edited.
7. HPO (Optuna, ASHA pruning) uses the val set only, 1 seed; final configs get
   3 seeds. Don't burn compute on multi-seed HPO trials.

## Environment constraints
- Server blocks system pip (PEP 668 / Ubuntu). Use `.venv-tools` venv for small
  utilities; conda envs (deimv2 / dfine / yolo11 — one per model, never merged)
  for training. `gdown` v5+: file id is positional, `--id` flag no longer exists.
- Long-running jobs (downloads, training) run inside tmux sessions.
- All commands run from repo root — scripts use relative paths.

## Workflow
- Dev on Mac (VSCode) → git push → git pull on GPU server → train there.
- Experiment tracking: Weights & Biases (`WANDB_API_KEY` in server-side `.env`,
  gitignored).
- Commit messages: conventional style (`feat:`, `fix:`, `docs:`, `chore:`).

## Key file map
- `PLAN.md` — full plan + Decision Log (single source of truth)
- `configs/data/pidray.yaml` — dataset paths, split params, augmentation policy
- `scripts/sync_data.sh` — data download (server, tmux)
- `src/data/validate_coco.py` — COCO integrity check + EDA stats
- `src/data/stratified_split.py` — 85/15 multi-label stratified split
- `src/data/sanity_check.py` — draws bboxes on 100 random images for eyeballing
- `src/data/preprocess.py` — CLAHE (clip=2.0, grid=8x8)
- `environments/*.yml` — one conda env spec per model
