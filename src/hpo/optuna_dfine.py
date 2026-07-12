"""
Optuna HPO study for D-FINE on PIDray (val set, single seed — per PLAN.md
Key Decisions: "tune on val ONLY; 1 seed during HPO, then 3 seeds on final
config only").

Same subprocess-per-trial design as src/hpo/optuna_deimv2.py (read that
docstring for the full rationale) — `epochs` fixed to the trial's full
budget (`MAX_EPOCH`) up front, not relaunched per ASHA rung, with
augmentation-schedule epoch milestones scaled proportionally to preserve
the paper's schedule *shape*.

Key differences from DEIMv2, verified against
third_party/D-FINE/configs/dfine/include/{dfine_hgnetv2,optimizer}.yml
2026-07-12 (do NOT assume DEIMv2's keys transfer without checking — the
CLAHE registry crash earlier this project taught that lesson):
  - Epoch count key is `epochs`, not `epoches` (DEIMv2's spelling).
  - Criterion class is `DFINECriterion`, not `DEIMCriterion` — same
    `weight_dict.loss_fgl` key name though (both default 0.15).
  - D-FINE has NO self-managed flatcosine scheduler (confirmed from the
    completed full 132-epoch run's log: `train_lr` sits constant at
    0.0001 from early epochs through epoch 131 — i.e. warmup then flat,
    no decay phase within the paper's own epoch budget). So there is no
    flat_epoch/no_aug_epoch to scale here, unlike DEIMv2.
  - Only ONE augmentation-schedule boundary exists (`policy.epoch` is a
    single int, not DEIMv2's 3-element list) — both `policy.epoch` and
    `collate_fn.stop_epoch` are pinned to the same milestone (120 at the
    paper's 132-epoch scale) and are scaled together here.

Search space (same 4 axes as DEIMv2, locked in PLAN.md Key Decisions):
  - optimizer.lr           (paper default 0.0002)
  - optimizer.weight_decay (paper default 0.0001)
  - lr_warmup_scheduler.warmup_duration (paper default 500, in iterations)
  - DFINECriterion.weight_dict.loss_fgl (paper default 0.15)
Backbone-specific lr (0.0001, half the general lr) is left untouched for
the same reason as DEIMv2 — see that script's docstring.

ASHA rungs: epoch 5 / 15 / 40 (min_resource=5, reduction_factor=3,
max_resource=40 — matches the scope agreed 2026-07-12).

Usage (run from repo root, inside the dfine conda env, inside tmux):
    conda activate dfine
    python src/hpo/optuna_dfine.py --n-trials 25
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
CONFIG = REPO_ROOT / "configs" / "model" / "dfine" / "dfine_pidray.yml"
TRAIN_SCRIPT = REPO_ROOT / "src" / "training" / "train_dfine.py"
STUDY_DB = REPO_ROOT / "results" / "optuna" / "dfine.db"

MAX_EPOCH = 40
RUNGS = [5, 15, MAX_EPOCH]
PAPER_TOTAL_EPOCH = 132
PAPER_AUG_STOP_EPOCH = 120  # both policy.epoch and collate_fn.stop_epoch

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

    aug_stop_epoch = _scale(PAPER_AUG_STOP_EPOCH)

    output_dir = REPO_ROOT / "outputs" / "hpo_dfine" / f"trial_{trial.number:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    updates = [
        f"epochs={MAX_EPOCH}",
        f"optimizer.lr={lr}",
        f"optimizer.weight_decay={weight_decay}",
        f"lr_warmup_scheduler.warmup_duration={warmup_duration}",
        f"DFINECriterion.weight_dict.loss_fgl={loss_fgl}",
        f"train_dataloader.dataset.transforms.policy.epoch={aug_stop_epoch}",
        f"train_dataloader.collate_fn.stop_epoch={aug_stop_epoch}",
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
                target_epoch_1idx = epoch + 1
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
        study_name="dfine_pidray",
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
