# PLAN.md — X-ray Detection Benchmark (Single Source of Truth)

> **วิธีใช้ไฟล์นี้**: ทุกครั้งที่เริ่มแชทใหม่กับ Claude ให้แนบไฟล์นี้ไปด้วยเพื่อให้ได้ context ครบ
> ทำขั้นตอนเสร็จ → ติ๊ก checkbox → commit
> ตัดสินใจอะไรใหม่ → บันทึกใน Decision Log ท้ายไฟล์ พร้อมวันที่และเหตุผล

**Last updated**: 2026-07-08
**Current phase**: Phase 2 (baselines) — 25-epoch reduced pass DONE for all 3 models (advisor check-in numbers ready); full paper-default run (132/132/100) still to come

---

## 1. Project Overview

**Goal**: Publication-quality benchmark of 3 object detectors on X-ray security imagery,
framed as AI research (not just engineering comparison).

**Models**: DEIMv2, D-FINE (DETR-based) vs YOLO11 (CNN one-stage)

**Primary dataset**: PIDray — public baggage X-ray, 12 classes, no access request needed
- Source: https://github.com/bywang2018/security-dataset (or HF: Voxel51/PIDray)
- 47,677 images; official train 29,457 / official test 18,220 (Easy 9,482 / Hard 3,773 / Hidden 5,005)
- Classes: gun, knife, wrench, pliers, scissors, hammer, handcuffs, baton, sprayer, powerbank, lighter, bullet
- License: academic/non-commercial only

**Cross-dataset targets (RQ3)**: OPIXray (email request: rstao@buaa.edu.cn), SIXray test set (bbox via SIXray-D, gated form at ROSE Lab NTU)

## 2. Research Questions

| RQ | Question | Data used |
|----|----------|-----------|
| RQ1 | DETR-based vs CNN one-stage on X-ray occlusion/overlap challenges | PIDray (esp. Hard/Hidden test) |
| RQ2 | Resolution handling / tiling effects on AP-small | PIDray (revisit SAHI need after EDA) |
| RQ3 | Cross-dataset domain gap within baggage X-ray | Train PIDray → test OPIXray / SIXray |
| RQ4 | Operational detection rate @ fixed false-alarm rate | PIDray Hidden subset = stress test |

## 3. Key Decisions (locked)

- ✅ **Split**: use PIDray OFFICIAL test split (Easy/Hard/Hidden) for literature comparability.
  Only official train (29,457) gets split by us: **train 85% / val 15%**, multi-label stratified, **seed=42**.
  Official test is NEVER touched until Phase 5.
- ✅ **Master annotation format**: single COCO JSON → converted per-model (YOLO txt for YOLO11).
- ✅ **Augmentation policy identical across all 3 models**: flip/crop OK, **NO color jitter / NO HSV**
  (X-ray domain). Must override each repo's defaults (e.g., disable YOLO11 HSV jitter).
- ✅ **Preprocessing**: CLAHE (clip=2.0, grid=8×8) — decide pre-compute vs on-the-fly after seeing data size.
- ✅ **HPO tool**: Optuna with ASHA/Hyperband pruning; ~20-30 trials/model; tune on val ONLY; 1 seed
  during HPO, then 3 seeds on final config only.
- ✅ **HPO search space**: DETR models: lr, weight decay, warmup epochs, aux loss weight.
  YOLO11: lr, box/cls loss weight ratio, mosaic prob, weight decay. Architecture params frozen.
- ✅ **Evaluation stack**: pycocotools (mAP@50-95, AP-S/M/L), TIDE (error types), ECE (calibration),
  detection rate @ fixed FAR (RQ4), paired bootstrap (significance), Pareto accuracy-vs-latency.
- ✅ **Workflow**: dev on Mac (VSCode) → GitHub → pull on server (GPU). Data synced OUTSIDE git.
  Experiment tracking: Weights & Biases. Separate conda env per model (3 envs).
- ✅ Report results separately per Easy/Hard/Hidden subset.

## 4. Phase Checklist

### Phase 0 — Setup (week 1)
- [x] 0.1 Create GitHub repo, push project skeleton
- [x] 0.2 Server: git clone
- [x] 0.3 Server: create 3 conda envs (deimv2 / dfine / yolo11) — `scripts/setup_env.sh`
- [x] 0.4 wandb account + API key on server (.env, gitignored)
- [ ] 0.5 (optional) VSCode Remote-SSH from Mac to server

### Phase 1 — Data (weeks 1-2)
- [x] 1.1 Download PIDray to server `data/raw/` (`scripts/sync_data.sh`; find real image link in repo README)
- [x] 1.2 EDA notebook: annotation format? image sizes? class distribution? official split structure?
      → decide: is SAHI tiling needed (RQ2)? pre-compute vs on-the-fly CLAHE?
- [x] 1.3 CONFIRMED: PIDray ships in COCO format (mmdet CocoDataset; xray_train.json + xray_test_{easy,hard,hidden}.json) → no converter needed, only run `src/data/validate_coco.py`
- [x] 1.4 Split official train → train/val (85/15, multi-label stratified, seed=42)
      → commit `data/splits/*.json` (image-id lists only — small files, git OK)
- [x] 1.5 Preprocessing + shared augmentation config
- [x] 1.6 **Sanity check (mandatory)**: visualize 100 random images with bboxes; verify COCO→YOLO
      conversion preserves boxes; count instances per class per split

### Phase 2 — Baselines (weeks 2-4)
- [x] **DONE (2026-07-08)**: reduced-epoch pass (25 epochs, all 3 models,
      down from paper-default 132/132/100) ran sequentially in tmux on the
      server — goal was a quick convergence/sanity check + numbers to show the
      advisor, NOT the official Phase 2 baseline record. Re-run at full
      paper-default epoch count still required before treating any mAP number
      here as final. Results (val set, AP@0.5:0.95 / AP@0.5):
      - YOLO11-S: 0.862 / 0.972 (~1 GPU-hour)
      - DEIMv2: 0.852 / 0.954 / AP@0.75=0.916 (several GPU-hours, grad accum)
      - D-FINE: 0.818 / 0.936 / AP@0.75=0.883 (several GPU-hours, grad accum)
      All 3 still improving epoch-over-epoch at epoch 24 (no plateau) — full
      paper-default run expected to raise all three further, especially
      AP@0.5:0.95 which has more headroom than the already-near-saturated
      AP@0.5.
- [ ] Train all 3 models with paper-default hyperparameters (no tuning)
  - **DEIMv2** (HGNetV2-S, `configs/model/deimv2/deimv2_pidray.yml`) — DONE:
    config wired, gradient accumulation (`src/training/train_deimv2.py`,
    `accum_micro_batch: 4`, verified 781 iters/epoch matching true batch=32
    math, max mem ~3.5GB vs OOM at 23GB+), and CLAHE wired
    (`src/data/clahe_transform.py`, backed by the same `albumentations.CLAHE`
    as `src/data/augmentation.py` — color-safe LAB-only enhancement, confirmed
    PIDray images are genuinely colorized dual-energy, not grayscale-as-RGB).
    All verified with smoke tests, loss finite throughout.
  - **D-FINE** (HGNetV2-S, `configs/model/dfine/dfine_pidray.yml`, based on
    `dfine_hgnetv2_s_coco.yml` paper defaults, not the unvalidated `_custom.yml`
    preset) — DONE, same as DEIMv2: gradient accumulation
    (`src/training/train_dfine.py`) + CLAHE wired, both verified. Two
    version-specific gotchas hit along the way: epoch override key is
    `epochs` not `epoches` (DEIMv2's spelling), and D-FINE's env resolved
    torchvision 0.27.1 (unpinned requirements.txt) vs DEIMv2's 0.20.1 —
    newer torchvision renamed the Transform hook `_transform`→`transform`,
    needed the same shim D-FINE's own `ConvertPILImage` already uses.
  - **YOLO11-S** (`configs/model/yolo11/{pidray_data,yolo11s_pidray}.yaml`,
    `src/training/train_yolo11.py`) — DONE: COCO→YOLO converter
    (`src/data/convert_coco_to_yolo.py`, symlinked images + generated labels,
    verified 8/8 sampled boxes correct — this also completed checklist 1.6),
    config wired, smoke test passed on the first try. No published paper
    defaults exist for YOLO11 (unlike DEIMv2/D-FINE) — ultralytics' own
    untouched defaults stand in for that role; only `hsv_h/s/v=0` overridden
    per hard rule #3. No third_party clone or gradient accumulation needed
    (pip package, far smaller memory footprint than the DETR decoders).
    CLAHE not yet wired for this model — ultralytics has its own optional
    Albumentations integration (different mechanism than DEIMv2/D-FINE's
    registry) that would need investigating separately if we want it applied.
  - **Phase 5 prep DONE**: `src/data/remap_test_categories.py` materializes a
    0-indexed copy of the official test JSONs (easy/hard/hidden) at
    `data/processed/pidray_test_{easy,hard,hidden}.json` — verified counts
    match `validate_coco.py` exactly (9482/9482, 3733/8892, 5005/5008).
    Original raw files under `data/raw/pidray/annotations/` untouched.
- [ ] Verify losses converge; record time/epoch → informs HPO budget
- [ ] Log baseline mAP as sanity floor
- [x] **Compare all 3 models' val-set results side by side in one table**
      (AP@0.5:0.95, AP@0.5, AP-S, per-model params/FLOPs) — done once after the
      25-epoch reduced pass (see table above, for the advisor check-in). Still
      need to repeat after the full paper-default (132/132/100) run. Neither
      table is the Phase 5 official comparison (that's on official test,
      post-HPO, 3 seeds) — this is a baseline sanity comparison only.

### Phase 3 — HPO (Optuna)
- [ ] Optuna study per model, ASHA pruning, val-set objective, SQLite storage for resume
- [ ] Lock best config per model

### Phase 4 — Final training (weeks 5-7)
- [ ] 3 seeds × best config × 3 models; save all checkpoints

### Phase 5 — Evaluation (weeks 8-10)
- [ ] Full metric suite on official test (per Easy/Hard/Hidden)
- [ ] Cross-dataset eval (RQ3) on OPIXray/SIXray — request access EARLY (long lead time)
- [ ] TIDE, ECE, FAR analysis, bootstrap significance, Pareto curves

### Phase 6 — Write-up (weeks 10-12)
- [ ] Paper draft, model cards, reproducibility package (env lockfiles, split files, configs)

## 5. Risks / Watch-outs

- OPIXray + SIXray-D access requests take time → **submit requests during Phase 1, not Phase 5**
- PIDray long-tail imbalance → may need class-weighted loss; check per-class AP early
- 3 repos have conflicting dependencies → never merge conda envs
- HPO compute vs 3-seed budget conflict → HPO with 1 seed only, 3 seeds only for finals

## 6. Decision Log

| Date | Decision | Reason |
|------|----------|--------|
| 2026-07-06 | Dropped cargo datasets (CargoXray/CargoX/SoC) | All gated/proprietary; dvc pull on CargoXray requires private Google Drive OAuth |
| 2026-07-06 | Dropped SIXray as primary | Detection bboxes (SIXray-D) require gated access request at ROSE Lab |
| 2026-07-06 | **PIDray = primary dataset** | Fully public, 12 classes, COCO-style bbox+masks, built-in Easy/Hard/Hidden test |
| 2026-07-06 | RQ3 reframed to within-baggage cross-dataset gap | No accessible cargo target domain |
| 2026-07-06 | Use official PIDray test split | Literature comparability; free difficulty breakdown |
| 2026-07-06 | Optuna chosen for HPO | Pruning support, resumable studies, PyTorch integration |
| 2026-07-06 | No COCO converter needed | Verified from official repo config: PIDray annotations are already COCO JSON |
| 2026-07-07 | CLAHE runs on-the-fly, not pre-computed | HPO (20-30 trials/model) needs to sweep clip/grid without re-processing ~47k images per trial; keeps disk usage flat |
| 2026-07-07 | Shared augmentation policy implemented via `albumentations` in `src/data/augmentation.py` | Cross-framework library that DEIMv2/D-FINE/YOLO11 dataloaders can all wrap, keeping one source of truth for CLAHE+flip+crop and enforcing no-color-jitter rule identically |
| 2026-07-07 | `materialize_split.py` remaps PIDray category_id (1-12) to 0-indexed labels (0-11) | DEIMv2 smoke test crashed with a CUDA "index out of bounds" assert — `CocoDetection` uses raw `category_id` as class label whenever `remap_mscoco_category=False`, so label 12 (bullet) indexed out of a 12-class head. **TODO before Phase 5**: apply the identical id-1 remap to a *separate* copy of the official test JSONs before feeding any model — never edit `data/raw/pidray/annotations/xray_test_*.json` directly. |
| 2026-07-07 | Gradient accumulation via custom entrypoints `src/training/train_{deimv2,dfine}.py`, not third_party edits | Paper `total_batch_size=32` assumes 8 GPUs; server has 1x RTX 3090/24GB (true batch=32 OOMs at multi-scale training res). Dataloader still yields real batches of 32 (LR schedule iteration math, Mosaic/MixUp/CopyBlend batch-level augmentation untouched) — the wrapper only chunks the GPU forward/backward into `accum_micro_batch`-sized pieces via a monkey-patched `train_one_epoch`, since editing `third_party/*/engine|src` directly would vanish on re-clone (gitignored). Config field `accum_micro_batch: 4` in both `deimv2_pidray.yml`/`dfine_pidray.yml`. |
| 2026-07-07 | **Bug fix**: `A.CLAHE(clip_limit=X)` treats a scalar as a `(1, X)` random range, not a fixed value | Discovered while wiring YOLO11's CLAHE — confirmed via `.get_params()` returning scattered values across `[1, 2]` instead of always `2.0`. CLAHE is a locked *preprocessing* decision (deterministic), not augmentation, so it must not vary per call. Fixed to `clip_limit=(2.0, 2.0)` in all three CLAHE call sites (`src/data/augmentation.py`, `src/data/clahe_transform.py` used by DEIMv2/D-FINE, `src/training/train_yolo11.py`). Re-verified all 3 models post-fix — no regressions, loss finite. |
| 2026-07-07 | Run all 3 baselines at 25 epochs first (not paper-default 132/132/100) | Need results to show the advisor soon; full paper-default epoch count is ~5-6 GPU-days sequential on the single RTX 3090. This reduced pass is NOT the official Phase 2 baseline record — re-run at full epoch count is still required before any number here is treated as final. |
| 2026-07-08 | 25-epoch reduced pass completed for all 3 models — val results: YOLO11-S 0.862/0.972, DEIMv2 0.852/0.954, D-FINE 0.818/0.936 (AP@0.5:0.95/AP@0.5) | All 3 still improving at epoch 24, no plateau. AP@0.5 is already near-saturated (lenient IoU=0.5 threshold + pretrained backbones + only 12 visually-distinct classes converge fast) while AP@0.5:0.95 (strict, averaged over IoU 0.5-0.95) has more headroom — expect full paper-default epoch run to raise AP@0.5:0.95 further without AP@0.5 moving much. Not yet conclusive for RQ1 (DETR vs CNN): YOLO11 currently leads but trained in ~1 GPU-hour vs several for DEIMv2/D-FINE, and none are at paper-default epoch count yet. |
