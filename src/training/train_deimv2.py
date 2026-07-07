"""
DEIMv2 training entrypoint — wraps third_party/DEIMv2/train.py to add
gradient accumulation, since the paper's total_batch_size=32 assumes 8 GPUs
and this server has one (RTX 3090, 24GB — a true batch of 32 at the
multi-scale training resolutions up to 800x800 OOMs).

Design: the dataloader still yields real batches of 32 exactly as the
config specifies — so per-epoch iteration count, the FlatCosineLRScheduler's
iteration math (warmup_iter, flat_epoch, no_aug_epoch), and batch-level
augmentation (Mosaic/MixUp/CopyBlend, which operate on the full batch in
BatchImageCollateFunction) all match the paper's design exactly, untouched.
The ONLY thing this wrapper changes is how a batch of 32 is pushed through
the GPU: split into micro-chunks of `accum_micro_batch` samples, forward
+ backward each chunk with loss scaled by 1/n_chunks (standard gradient
accumulation convention — same approach used by HuggingFace/timm/Lightning),
gradients accumulate across chunks, and optimizer.step()/ema.update()/
lr_scheduler.step() still fire exactly once per batch of 32, same as
unmodified DEIMv2. Everything else in the loop (NaN guard, logging,
tensorboard) is copied from third_party/DEIMv2/engine/solver/det_engine.py
verbatim, only the per-batch GPU work is chunked.

This can't be a change to third_party/DEIMv2/engine/solver/det_engine.py
directly: that directory is gitignored (cloned fresh per machine by
scripts/setup_env.sh), so any edit there would silently vanish on re-clone.
Monkey-patching from our own git-tracked entrypoint is the only way to make
this change durable.

Usage (run from repo root):
    torchrun --nproc_per_node=1 src/training/train_deimv2.py \
        -c configs/model/deimv2/deimv2_pidray.yml \
        -u accum_micro_batch=4
"""
import argparse
import math
import sys
from pathlib import Path

DEIMV2_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "DEIMv2"
sys.path.insert(0, str(DEIMV2_ROOT))

import torch  # noqa: E402

from engine.core import YAMLConfig, yaml_utils  # noqa: E402
from engine.misc import MetricLogger, SmoothedValue, dist_utils  # noqa: E402
from engine.solver import TASKS  # noqa: E402
import engine.solver.det_solver as det_solver_module  # noqa: E402


def build_accum_train_one_epoch(micro_batch_size: int):
    """Drop-in replacement for det_engine.train_one_epoch with gradient accumulation."""

    def accum_train_one_epoch(self_lr_scheduler, lr_scheduler, model, criterion,
                               data_loader, optimizer, device, epoch, max_norm=0, **kwargs):
        model.train()
        criterion.train()
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)

        print_freq = kwargs.get('print_freq', 10)
        writer = kwargs.get('writer', None)
        ema = kwargs.get('ema', None)
        scaler = kwargs.get('scaler', None)
        lr_warmup_scheduler = kwargs.get('lr_warmup_scheduler', None)

        cur_iters = epoch * len(data_loader)

        for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            global_step = epoch * len(data_loader) + i
            metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

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

                if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
                    print(outputs['pred_boxes'])
                    state = model.state_dict()
                    new_state = {k.replace('module.', ''): v for k, v in state.items()}
                    dist_utils.save_on_master({'model': new_state}, "./NaN.pth")

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

            if self_lr_scheduler:
                optimizer = lr_scheduler.step(cur_iters + i, optimizer)
            else:
                if lr_warmup_scheduler is not None:
                    lr_warmup_scheduler.step()

            loss_dict_reduced = dist_utils.reduce_dict(loss_dict_sum)
            loss_value = sum(loss_dict_reduced.values())

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            metric_logger.update(loss=loss_value, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            if writer and dist_utils.is_main_process() and global_step % 10 == 0:
                writer.add_scalar('Loss/total', loss_value.item(), global_step)
                for j, pg in enumerate(optimizer.param_groups):
                    writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
                for k, v in loss_dict_reduced.items():
                    writer.add_scalar(f'Loss/{k}', v.item(), global_step)

        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return accum_train_one_epoch


def main(args) -> None:
    from engine.misc import dist_utils as _dist_utils
    _dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

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

    _dist_utils.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='')
    parser.add_argument('-r', '--resume', type=str, help='resume from checkpoint')
    parser.add_argument('-t', '--tuning', type=str, help='tuning from checkpoint')
    parser.add_argument('-d', '--device', type=str, help='device')
    parser.add_argument('--seed', type=int, default=0, help='exp reproducibility')
    parser.add_argument('--use-amp', action='store_true', help='auto mixed precision training')
    parser.add_argument('--output-dir', type=str, help='output directoy')
    parser.add_argument('--summary-dir', type=str, help='tensorboard summry')
    parser.add_argument('--test-only', action='store_true', default=False)
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')
    parser.add_argument('--accum-micro-batch', type=int, default=None,
                         help='max samples per GPU forward/backward pass; batches larger '
                              'than this accumulate gradients over multiple chunks so the '
                              'effective batch size (and LR schedule) matches the config '
                              'total_batch_size exactly. Omit to disable (single forward pass).')
    parser.add_argument('--print-method', type=str, default='builtin', help='print method')
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')
    parser.add_argument('--local-rank', type=int, help='local rank id')
    args = parser.parse_args()

    main(args)
