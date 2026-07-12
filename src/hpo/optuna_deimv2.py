"""
Optuna HPO study for DEIMv2 on PIDray (val set, single seed — per PLAN.md
Key Decisions: "tune on val ONLY; 1 seed during HPO, then 3 seeds on final
config only").

Design note — why this drives training via ONE long-lived subprocess per
trial instead of re-launching per ASHA rung: DEIMv2's `epoches` value
determines the *shape* of its self-managed flatcosine LR schedule
(flat_epoch/no_aug_epoch are absolute epoch numbers, not fractions) and its
augmentation-removal schedule (Mosaic/MixUp/CopyBlend stop epochs). If a
trial's `epoches` changed between rungs (e.g. 5 -> 15 -> 40, relaunched with
`-r` resume each time), the model would be trained under a *different*
schedule shape at each rung relative to what it already learned under —
not a faithful "continue this run" resume. Instead: `epoches` is fixed to
the trial's full budget (`MAX_EPOCH`) up front, with `flat_epoch`/
`no_aug_epoch`/augmentation-schedule epoch milestones all scaled down
proportionally to preserve the *shape* of the paper's schedule (e.g.
flat_epoch=64 at 132 total -> ~19 at 40 total) — see PLAN.md Decision Log
for the reasoning. ASHA pruning then means "watch this one continuous run's
log.txt and kill it early if it's not competitive", not "restart with a
different total".

Search space (locked in PLAN.md Key Decisions — DETR models: lr, weight
decay, warmup epochs, aux loss weight; architecture params frozen):
  - optimizer.lr           (paper default 0.0004)
  - optimizer.weight_decay (paper default 0.0001)
  - lr_warmup_scheduler.warmup_duration (paper default 500, in iterations)
  - DEIMCriterion.weight_dict.loss_fgl  (paper default 0.15 — the
    fine-grained-localization auxiliary loss weight)
Only the *shared* (non-backbone-specific) lr is tuned — DEIMv2's optimizer
config gives the backbone param group a separately fixed lr (half of the
overall lr in the paper default); the `-u` override mechanism only cleanly
supports dict-nested keys (`a.b=x`), not indexing into the `optimizer.params`
list, so the backbone-specific entry is left at its paper-default value for
every trial. This is a scoped simplification, not a full re-derivation of
the paper's differential-lr ratio.

ASHA rungs: epoch 5 / 15 / 40 (min_resource=5, reduction_factor=3,
max_resource=40 — matches the scope agreed 2026-07-12).

Usage (run from repo root, inside the deimv2 conda env, inside tmux):
    conda activate deimv2
    python src/hpo/optuna_deimv2.py --n-trials 25

Resume an interrupted study (same command — SQLite storage persists trials):
    python src/hpo/optuna_deimv2.py --n-trials 25
"""
import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import optuna

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs" / "model" / "deimv2" / "deimv2_pidray.yml"
TRAIN_SCRIPT = REPO_ROOT / "src" / "training" / "train_deimv2.py"
STUDY_DB = REPO_ROOT / "results" / "optuna" / "deimv2.db"

MAX_EPOCH = 40
RUNGS = [5, 15, MAX_EPOCH]  # ASHA report/prune checkpoints
# Paper defaults (132-epoch schedule) these milestones are scaled from:
PAPER_TOTAL_EPOCH = 132
PAPER_FLAT_EPOCH = 64
PAPER_NO_AUG_EPOCH = 12
PAPER_POLICY_EPOCHS = [4, 64, 120]
PAPER_MIXUP_EPOCHS = [4, 64]
PAPER_STOP_EPOCH = 120
PAPER_COPYBLEND_EPOCHS = [4, 120]

POLL_INTERVAL_S = 30


def _scale(epoch: int) -> int:
    return max(1, round(epoch * MAX_EPOCH / PAPER_TOTAL_EPOCH))


def read_log(output_dir: Path) -> list[dict]:
    log_path = output_dir / "log.txt"
    if not log_path.exists():
        return []
    lines = log_path.read_text().strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def kill_process_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def objective(trial: optuna.Trial) -> float:
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    warmup_duration = trial.suggest_int("warmup_duration", 250, 1500, step=250)
    loss_fgl = trial.suggest_float("loss_fgl", 0.05, 0.5, log=True)

    flat_epoch = _scale(PAPER_FLAT_EPOCH)
    no_aug_epoch = max(1, round(PAPER_NO_AUG_EPOCH * MAX_EPOCH / PAPER_TOTAL_EPOCH))
    policy_epochs = [_scale(e) for e in PAPER_POLICY_EPOCHS]
    mixup_epochs = [_scale(e) for e in PAPER_MIXUP_EPOCHS]
    stop_epoch = _scale(PAPER_STOP_EPOCH)
    copyblend_epochs = [_scale(e) for e in PAPER_COPYBLEND_EPOCHS]

    output_dir = REPO_ROOT / "outputs" / "hpo_deimv2" / f"trial_{trial.number:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    updates = [
        f"epoches={MAX_EPOCH}",
        f"optimizer.lr={lr}",
        f"optimizer.weight_decay={weight_decay}",
        f"lr_warmup_scheduler.warmup_duration={warmup_duration}",
        f"DEIMCriterion.weight_dict.loss_fgl={loss_fgl}",
        f"flat_epoch={flat_epoch}",
        f"no_aug_epoch={no_aug_epoch}",
        f"train_dataloader.dataset.transforms.policy.epoch={policy_epochs}",
        f"train_dataloader.collate_fn.mixup_epochs={mixup_epochs}",
        f"train_dataloader.collate_fn.stop_epoch={stop_epoch}",
        f"train_dataloader.collate_fn.copyblend_epochs={copyblend_epochs}",
    ]

    cmd = [
        "torchrun", "--nproc_per_node=1", str(TRAIN_SCRIPT),
        "-c", str(CONFIG),
        "--seed", "0",
        "--output-dir", str(output_dir),
        "-u", *updates,
    ]

    stdout_path = output_dir / "stdout.log"
    with open(stdout_path, "w") as stdout_f:
        proc = subprocess.Popen(
            cmd, cwd=REPO_ROOT, stdout=stdout_f, stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    print(f"[trial {trial.number}] launched pid={proc.pid} "
          f"lr={lr:.2e} wd={weight_decay:.2e} warmup={warmup_duration} "
          f"loss_fgl={loss_fgl:.3f} -> {output_dir}")

    reported_epochs: set[int] = set()
    last_ap = 0.0
    try:
        while True:
            retcode = proc.poll()
            entries = read_log(output_dir)
            for entry in entries:
                epoch = entry["epoch"]
                if epoch in reported_epochs:
                    continue
                target_epoch_1idx = epoch + 1  # log.txt epoch is 0-indexed
                if target_epoch_1idx not in RUNGS:
                    continue
                ap = entry["test_coco_eval_bbox"][0]
                last_ap = ap
                reported_epochs.add(epoch)
                trial.report(ap, step=target_epoch_1idx)
                print(f"[trial {trial.number}] epoch {target_epoch_1idx}: AP@0.5:0.95={ap:.4f}")
                if trial.should_prune():
                    print(f"[trial {trial.number}] pruned at epoch {target_epoch_1idx}")
                    kill_process_group(proc)
                    raise optuna.TrialPruned()

            if retcode is not None:
                break
            time.sleep(POLL_INTERVAL_S)

        if retcode != 0:
            print(f"[trial {trial.number}] training process exited with code {retcode} "
                  f"(see {stdout_path}) — treating as failed trial")
            raise optuna.TrialPruned()

        entries = read_log(output_dir)
        if entries:
            last_ap = entries[-1]["test_coco_eval_bbox"][0]
        return last_ap
    finally:
        if proc.poll() is None:
            kill_process_group(proc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=25)
    args = parser.parse_args()

    STUDY_DB.parent.mkdir(parents=True, exist_ok=True)
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=RUNGS[0], reduction_factor=3,
    )
    study = optuna.create_study(
        study_name="deimv2_pidray",
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
