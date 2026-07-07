"""
D-FINE training entrypoint — same gradient-accumulation wrapper approach as
src/training/train_deimv2.py (read that docstring for the full rationale).
D-FINE's train_one_epoch has a different signature (no self-managed
FlatCosineLRScheduler — D-FINE steps a standard MultiStepLR once per epoch
in fit(), untouched by this wrapper) but the same core idea: split each
already-batch-of-32 into micro-chunks, accumulate gradients, one
optimizer.step()/ema.update() per batch of 32.

NAMING COLLISION WARNING: D-FINE's own package is named `src` (this repo's
src/solver, src/core, etc.), the exact same top-level name as OUR project's
src/. This works today because this script never imports anything from our
own src/ package in the same process — if a future change (e.g. wiring
src/data/augmentation.py's CLAHE policy) needs to import our own src.*
from this file, do it via importlib with an explicit file path, not a bare
`from src... import ...`, or it will silently resolve to D-FINE's package
instead (sys.path[0] wins, whichever `src` got imported first is cached in
sys.modules under that name).

Usage (run from repo root):
    torchrun --nproc_per_node=1 src/training/train_dfine.py \
        -c configs/model/dfine/dfine_pidray.yml \
        -u accum_micro_batch=4
"""
import argparse
import math
import sys
from pathlib import Path

DFINE_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "D-FINE"
sys.path.insert(0, str(DFINE_ROOT))

import torch  # noqa: E402

from src.core import YAMLConfig, yaml_utils  # noqa: E402
from src.misc import MetricLogger, SmoothedValue, dist_utils, save_samples  # noqa: E402
from src.solver import TASKS  # noqa: E402
import src.solver.det_solver as det_solver_module  # noqa: E402


def build_accum_train_one_epoch(micro_batch_size: int):
    """Drop-in replacement for det_engine.train_one_epoch with gradient accumulation."""

    def accum_train_one_epoch(model, criterion, data_loader, optimizer, device, epoch,
                               use_wandb=False, max_norm=0, **kwargs):
        if use_wandb:
            import wandb

        model.train()
        criterion.train()
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

        epochs = kwargs.get("epochs", None)
        header = "Epoch: [{}]".format(epoch) if epochs is None else "Epoch: [{}/{}]".format(epoch, epochs)

        print_freq = kwargs.get("print_freq", 10)
        writer = kwargs.get("writer", None)
        ema = kwargs.get("ema", None)
        scaler = kwargs.get("scaler", None)
        lr_warmup_scheduler = kwargs.get("lr_warmup_scheduler", None)
        losses = []

        output_dir = kwargs.get("output_dir", None)
        num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)

        for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            global_step = epoch * len(data_loader) + i
            metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

            if global_step < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():
                save_samples(samples, targets, output_dir, "train", normalized=True, box_fmt="cxcywh")

            samples = samples.to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            batch_size = samples.shape[0]
            n_chunks = math.ceil(batch_size / micro_batch_size)

            optimizer.zero_grad()
            loss_dict_sum = None
            for c in range(n_chunks):
                lo, hi = c * micro_batch_size, min((c + 1) * micro_batch_size, batch_size)
                chunk_samples = samples[lo:hi]
                chunk_targets = targets[lo:hi]

                if scaler is not None:
                    with torch.autocast(device_type=str(device), cache_enabled=True):
                        outputs = model(chunk_samples, targets=chunk_targets)
                else:
                    outputs = model(chunk_samples, targets=chunk_targets)

                if torch.isnan(outputs["pred_boxes"]).any() or torch.isinf(outputs["pred_boxes"]).any():
                    print(outputs["pred_boxes"])
                    state = model.state_dict()
                    new_state = {k.replace("module.", ""): v for k, v in state.items()}
                    dist_utils.save_on_master({"model": new_state}, "./NaN.pth")

                if scaler is not None:
                    with torch.autocast(device_type=str(device), enabled=False):
                        loss_dict = criterion(outputs, chunk_targets, **metas)
                    loss = sum(loss_dict.values()) / n_chunks
                    scaler.scale(loss).backward()
                else:
                    loss_dict = criterion(outputs, chunk_targets, **metas)
                    loss = sum(loss_dict.values()) / n_chunks
                    loss.backward()

                if loss_dict_sum is None:
                    loss_dict_sum = {k: v.detach() / n_chunks for k, v in loss_dict.items()}
                else:
                    for k, v in loss_dict.items():
                        loss_dict_sum[k] = loss_dict_sum[k] + v.detach() / n_chunks

            if max_norm > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if ema is not None:
                ema.update(model)

            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

            loss_dict_reduced = dist_utils.reduce_dict(loss_dict_sum)
            loss_value = sum(loss_dict_reduced.values())
            losses.append(loss_value.detach().cpu().numpy())

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            metric_logger.update(loss=loss_value, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            if writer and dist_utils.is_main_process() and global_step % 10 == 0:
                writer.add_scalar("Loss/total", loss_value.item(), global_step)
                for j, pg in enumerate(optimizer.param_groups):
                    writer.add_scalar(f"Lr/pg_{j}", pg["lr"], global_step)
                for k, v in loss_dict_reduced.items():
                    writer.add_scalar(f"Loss/{k}", v.item(), global_step)

        if use_wandb:
            import numpy as np
            wandb.log({"lr": optimizer.param_groups[0]["lr"], "epoch": epoch, "train/loss": np.mean(losses)})

        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return accum_train_one_epoch


def main(args) -> None:
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items()
                         if k not in ['update', 'accum_micro_batch'] and v is not None})

    cfg = YAMLConfig(args.config, **update_dict)

    if args.resume or args.tuning:
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    micro_batch = args.accum_micro_batch or cfg.yaml_cfg.get('accum_micro_batch')
    if micro_batch:
        total_batch = cfg.yaml_cfg['train_dataloader']['total_batch_size']
        n_chunks = math.ceil(total_batch / micro_batch)
        print(f"[grad-accum] total_batch_size={total_batch}, "
              f"micro_batch={micro_batch} -> {n_chunks} accumulation steps/iter")
        det_solver_module.train_one_epoch = build_accum_train_one_epoch(micro_batch)

    print('cfg: ', cfg.__dict__)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, help='resume from checkpoint')
    parser.add_argument('-t', '--tuning', type=str, help='tuning from checkpoint')
    parser.add_argument('-d', '--device', type=str, help='device')
    parser.add_argument('--seed', type=int, help='exp reproducibility')
    parser.add_argument('--use-amp', action='store_true', help='auto mixed precision training')
    parser.add_argument('--output-dir', type=str, help='output directoy')
    parser.add_argument('--summary-dir', type=str, help='tensorboard summry')
    parser.add_argument('--test-only', action='store_true', default=False)
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')
    parser.add_argument('--accum-micro-batch', type=int, default=None,
                         help='max samples per GPU forward/backward pass; batches larger '
                              'than this accumulate gradients over multiple chunks so the '
                              'effective batch size matches the config total_batch_size '
                              'exactly. Omit to disable (single forward pass).')
    parser.add_argument('--print-method', type=str, default='builtin', help='print method')
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')
    parser.add_argument('--local-rank', type=int, help='local rank id')
    args = parser.parse_args()

    main(args)
