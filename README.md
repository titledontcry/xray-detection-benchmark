# X-ray Cargo/Baggage Object Detection Benchmark
Rigorous benchmarking of DEIMv2, D-FINE, and YOLO11 on PIDray (baggage X-ray security dataset).

## Project structure

```
xray-benchmark/
├── configs/                # Hydra/YAML configs (git-tracked, small)
│   ├── data/                #   dataset paths, split ratios, class maps
│   ├── model/                #   per-architecture hyperparameters
│   └── training/            #   seeds, epochs, optimizer settings
├── data/
│   ├── splits/               #   train/val/test image-id lists (git-tracked, small JSON/CSV)
│   ├── raw/                  #   [NOT in git] raw PIDray download
│   └── processed/            #   [NOT in git] COCO JSON + preprocessed images
├── src/
│   ├── data/                 #   COCO conversion, stratified split, CLAHE preprocessing
│   ├── training/              #   train entrypoints per model
│   ├── evaluation/            #   pycocotools eval, TIDE, ECE, bootstrap significance
│   └── utils/
├── scripts/                 # shell scripts: env setup, data sync, launch training
├── environments/            # one conda/venv spec PER model (avoids dependency conflicts)
├── notebooks/                # exploratory analysis only — no pipeline logic here
└── .github/workflows/        # CI: lint + config validation on push
```

## Environments — one per model (they have incompatible dependency trees)

```bash
# On the SERVER (not on Mac — these need CUDA):
conda env create -f environments/deimv2.yml
conda env create -f environments/dfine.yml
conda env create -f environments/yolo11.yml
```

## Data workflow (NEVER goes through git)

```bash
# On server, one-time:
bash scripts/sync_data.sh   # rsync/rclone pull of PIDray from wherever it's hosted

# After preprocessing runs (server-side), only the SPLIT FILES (small JSON)
# get committed back to git so Mac + server always agree on which image
# belongs to train/val/test:
git add data/splits/*.json
git commit -m "chore: update stratified split manifest"
```

## Day-to-day loop (Mac <-> Server)

```bash
# On Mac: write code, edit configs
git add -A && git commit -m "feat: add CLAHE preprocessing step"
git push origin main

# On server:
git pull origin main
conda activate deimv2
python src/training/train_deimv2.py --config configs/model/deimv2.yaml
```

## Experiment tracking

All runs log to Weights & Biases (`wandb`) so you can monitor from the Mac
browser without SSH-ing into the server. Set `WANDB_API_KEY` in `.env`
(gitignored) on the server only.

## Reproducibility checklist (fill in as project progresses)

- [ ] `environments/*.yml` lockfiles committed
- [ ] `data/splits/*.json` committed (vehicle/image-id level, not raw images)
- [ ] Docker image tag or conda env hash recorded per experiment
- [ ] Random seeds fixed and logged (3 seeds per final config)
- [ ] Model cards written for each of the 3 architectures
