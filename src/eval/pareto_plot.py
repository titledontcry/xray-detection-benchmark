"""
Accuracy-vs-FLOPs Pareto scatter plot — the standard comparison chart in
object-detection efficiency papers (confirmed convention: DEIMv2's own repo
ships third_party/DEIMv2/figures/deimv2_coco_AP_vs_GFLOPs.png). Directly
answers RQ1 framed as a cost/accuracy tradeoff, not just "which number is
bigger": AP@0.5:0.95 on the Y axis, GFLOPs on the X axis, one point per
model.

Reads model stats from a small JSON file (results/model_stats.json) rather
than hardcoding numbers here, since AP/FLOPs get updated as training
progresses (25-epoch pass now, full paper-default epoch later) — keeps this
script reusable across both without editing code.

model_stats.json format:
    {
      "DEIMv2":   {"ap": 0.852, "flops_g": 12.3, "params_m": 14.1},
      "D-FINE":   {"ap": 0.818, "flops_g": 11.9, "params_m": 10.19},
      "YOLO11-S": {"ap": 0.862, "flops_g": 21.3, "params_m": 9.42}
    }

Usage:
    python src/eval/pareto_plot.py \
        --stats results/model_stats.json \
        --out results/pareto_ap_vs_flops_25ep.png \
        --title "PIDray val — 25-epoch reduced pass"
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", type=str, default="AP@0.5:0.95 vs FLOPs")
    args = parser.parse_args()

    with open(args.stats) as f:
        stats = json.load(f)

    missing = [name for name, s in stats.items() if s.get("ap") is None or s.get("flops_g") is None]
    if missing:
        print(f"[warn] skipping models with missing ap/flops_g: {missing}")

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, s in stats.items():
        if s.get("ap") is None or s.get("flops_g") is None:
            continue
        ax.scatter(s["flops_g"], s["ap"] * 100, s=80, zorder=3)
        label = name
        if s.get("params_m") is not None:
            label += f"\n({s['params_m']:.1f}M params)"
        ax.annotate(label, (s["flops_g"], s["ap"] * 100),
                    textcoords="offset points", xytext=(8, 6), fontsize=9)

    ax.set_xlabel("GFLOPs (lower = cheaper)")
    ax.set_ylabel("AP@0.5:0.95 (%, higher = better)")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3, zorder=0)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"[pareto_plot] saved to {args.out}")


if __name__ == "__main__":
    main()
