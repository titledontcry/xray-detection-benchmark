"""
Optuna HPO study for YOLO11-S on PIDray (val set, single seed — per
PLAN.md Key Decisions: "tune on val ONLY; 1 seed during HPO, then 3 seeds
on final config only").

Unlike DEIMv2/D-FINE (src/hpo/optuna_{deimv2,dfine}.py — subprocess-per-
trial, polling log.txt, since those don't expose an in-process training
hook we can trust), YOLO11 runs `model.train()` directly in this process:
ultralytics exposes an `on_fit_epoch_end` callback (confirmed available on
the installed version, 2026-07-12) fired after every epoch's validation,
which is used to report to Optuna and — by setting `trainer.stop = True`,
the same mechanism ultralytics' own EarlyStopping callback uses — cleanly
break out of `model.train()`'s internal loop when a trial should be
pruned, without needing to manage a subprocess.

Search space (locked in PLAN.md Key Decisions — YOLO11: lr, box/cls loss
weight ratio, mosaic prob, weight decay):
  - lr0           (ultralytics default 0.01)
  - weight_decay  (ultralytics default 0.0005)
  - box           (box loss gain, default 7.5)
  - cls           (cls loss gain, default 0.5) — box+cls together capture
    the "box/cls loss weight ratio" from the locked search space
  - mosaic        (mosaic aug probability, default 1.0)

`close_mosaic` (epochs before the end with mosaic disabled — ultralytics
default 10, i.e. last 10% of a 100-epoch run) is scaled proportionally to
the trial's epoch budget, same rationale as DEIMv2/D-FINE's augmentation-
schedule scaling (see optuna_deimv2.py's docstring) — otherwise a 30-epoch
trial would never reach the "mosaic off" phase the final 100-epoch run
spends its last 10 epochs in.

ASHA rungs: epoch 5 / 15 / 30 (min_resource=5, reduction_factor=3,
max_resource=30 — 30/100 matches DEIMv2/D-FINE's 40/132 proportion, per
the scope agreed 2026-07-12).

Usage (run from repo root, inside the yolo11 conda env, inside tmux):
    conda activate yolo11
    python src/hpo/optuna_yolo11.py --n-trials 25
"""
import argparse
from pathlib import Path

import albumentations as A
import optuna
import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs" / "model" / "yolo11" / "yolo11s_pidray.yaml"
STUDY_DB = REPO_ROOT / "results" / "optuna" / "yolo11.db"

MAX_EPOCH = 30
RUNGS = [5, 15, MAX_EPOCH]
PAPER_TOTAL_EPOCH = 100
DEFAULT_CLOSE_MOSAIC = 10


def find_ap_key(metrics: dict) -> str | None:
    return next((k for k in metrics if "mAP50-95" in k), None)


def objective(trial: optuna.Trial) -> float:
    lr0 = trial.suggest_float("lr0", 1e-3, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    box = trial.suggest_float("box", 2.0, 15.0)
    cls = trial.suggest_float("cls", 0.2, 2.0)
    mosaic = trial.suggest_float("mosaic", 0.0, 1.0)

    close_mosaic = max(1, round(MAX_EPOCH * DEFAULT_CLOSE_MOSAIC / PAPER_TOTAL_EPOCH))

    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)

    cfg["epochs"] = MAX_EPOCH
    cfg["close_mosaic"] = close_mosaic
    cfg["lr0"] = lr0
    cfg["weight_decay"] = weight_decay
    cfg["box"] = box
    cfg["cls"] = cls
    cfg["mosaic"] = mosaic
    cfg["seed"] = 0
    cfg["name"] = f"hpo_yolo11/trial_{trial.number:03d}"
    # Same fixed clip/grid CLAHE injection as src/training/train_yolo11.py —
    # one CLAHE definition shared across all 3 models (CLAUDE.md hard rule #4).
    cfg["augmentations"] = [A.CLAHE(clip_limit=(2.0, 2.0), tile_grid_size=(8, 8), p=1.0)]

    model_name = cfg.pop("model")
    model = YOLO(model_name)

    print(f"[trial {trial.number}] lr0={lr0:.2e} wd={weight_decay:.2e} "
          f"box={box:.2f} cls={cls:.2f} mosaic={mosaic:.2f} "
          f"close_mosaic={close_mosaic}")

    state = {"pruned": False, "reported": set()}

    def on_fit_epoch_end(trainer):
        epoch = trainer.epoch + 1  # trainer.epoch is 0-indexed
        if epoch not in RUNGS or epoch in state["reported"]:
            return
        state["reported"].add(epoch)
        ap_key = find_ap_key(trainer.metrics)
        if ap_key is None:
            print(f"[trial {trial.number}] WARNING: no mAP50-95 key in "
                  f"trainer.metrics ({list(trainer.metrics.keys())}) — skipping report")
            return
        ap = trainer.metrics[ap_key]
        trial.report(ap, step=epoch)
        print(f"[trial {trial.number}] epoch {epoch}: AP@0.5:0.95={ap:.4f}")
        if trial.should_prune():
            print(f"[trial {trial.number}] pruned at epoch {epoch}")
            state["pruned"] = True
            trainer.stop = True

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    results = model.train(**cfg)

    if state["pruned"]:
        raise optuna.TrialPruned()

    final_ap = None
    if results is not None and hasattr(results, "results_dict"):
        ap_key = find_ap_key(results.results_dict)
        if ap_key is not None:
            final_ap = results.results_dict[ap_key]
    if final_ap is None:
        raise RuntimeError(
            f"[trial {trial.number}] could not find final AP@0.5:0.95 in "
            f"results.results_dict — inspect manually before trusting this study"
        )
    return final_ap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=25)
    args = parser.parse_args()

    STUDY_DB.parent.mkdir(parents=True, exist_ok=True)
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=RUNGS[0], reduction_factor=3,
    )
    study = optuna.create_study(
        study_name="yolo11_pidray",
        storage=f"sqlite:///{STUDY_DB}",
        direction="maximize",
        pruner=pruner,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.n_trials)

    print("\n=== Best trial ===")
    print(f"AP@0.5:0.95 = {study.best_value:.4f}")
    print(f"params = {study.best_params}")


if __name__ == "__main__":
    main()
