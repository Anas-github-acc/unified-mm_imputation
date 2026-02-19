#!/usr/bin/env python3
"""
Compare Baseline vs Optimized (N4) results.

Loads evaluation metrics from both experiments and produces:
  - Comparison table
  - Bar chart comparison
  - Side-by-side visual comparison figure

Usage:
  python compare_results.py --baseline_results ./results/baseline/metrics.json \
                             --optimized_results ./results/optimized/metrics.json
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(path):
    """Load metrics JSON."""
    with open(path) as f:
        return json.load(f)


def print_comparison_table(baseline, optimized):
    """Print formatted comparison table."""
    print("\n" + "=" * 75)
    print("COMPARISON: Baseline vs Optimized (N4 Bias Correction)")
    print("=" * 75)

    # Overall comparison
    b_overall = baseline["overall"]
    o_overall = optimized["overall"]

    print(f"\n{'Method':<30} {'PSNR (dB)':>12} {'SSIM':>12}")
    print(f"{'-'*55}")
    print(f"{'Baseline GAN':<30} {b_overall['psnr_mean']:>12.2f} {b_overall['ssim_mean']:>12.4f}")
    print(f"{'+ N4 Bias Correction':<30} {o_overall['psnr_mean']:>12.2f} {o_overall['ssim_mean']:>12.4f}")
    print(f"{'-'*55}")

    psnr_diff = o_overall['psnr_mean'] - b_overall['psnr_mean']
    ssim_diff = o_overall['ssim_mean'] - b_overall['ssim_mean']
    print(f"{'Improvement':<30} {psnr_diff:>+12.2f} {ssim_diff:>+12.4f}")
    print()

    # Per-scenario comparison
    print(f"\n{'Scenario':<12} {'Baseline PSNR':>14} {'Optimized PSNR':>15} {'Delta':>8} "
          f"{'Baseline SSIM':>14} {'Optimized SSIM':>15} {'Delta':>8}")
    print(f"{'-'*90}")

    b_scenarios = baseline.get("per_scenario", {})
    o_scenarios = optimized.get("per_scenario", {})

    for key in sorted(b_scenarios.keys()):
        if key in o_scenarios:
            bp = b_scenarios[key]["psnr_mean"]
            op = o_scenarios[key]["psnr_mean"]
            bs = b_scenarios[key]["ssim_mean"]
            os_ = o_scenarios[key]["ssim_mean"]

            print(f"{key:<12} {bp:>14.2f} {op:>15.2f} {op-bp:>+8.2f} "
                  f"{bs:>14.4f} {os_:>15.4f} {os_-bs:>+8.4f}")

    print("=" * 75)


def create_comparison_chart(baseline, optimized, output_path):
    """Create bar chart comparing baseline vs optimized."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    b_scenarios = baseline.get("per_scenario", {})
    o_scenarios = optimized.get("per_scenario", {})

    scenarios = sorted(set(b_scenarios.keys()) & set(o_scenarios.keys()))

    x = np.arange(len(scenarios))
    width = 0.35

    # PSNR comparison
    b_psnr = [b_scenarios[s]["psnr_mean"] for s in scenarios]
    o_psnr = [o_scenarios[s]["psnr_mean"] for s in scenarios]

    bars1 = ax1.bar(x - width/2, b_psnr, width, label="Baseline", color="#2196F3", alpha=0.8)
    bars2 = ax1.bar(x + width/2, o_psnr, width, label="+ N4 Correction", color="#4CAF50", alpha=0.8)

    ax1.set_xlabel("Scenario")
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title("PSNR Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, rotation=45)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # SSIM comparison
    b_ssim = [b_scenarios[s]["ssim_mean"] for s in scenarios]
    o_ssim = [o_scenarios[s]["ssim_mean"] for s in scenarios]

    ax2.bar(x - width/2, b_ssim, width, label="Baseline", color="#2196F3", alpha=0.8)
    ax2.bar(x + width/2, o_ssim, width, label="+ N4 Correction", color="#4CAF50", alpha=0.8)

    ax2.set_xlabel("Scenario")
    ax2.set_ylabel("SSIM")
    ax2.set_title("SSIM Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, rotation=45)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # Add overall comparison text
    b_ov = baseline["overall"]
    o_ov = optimized["overall"]
    fig.text(
        0.5, 0.01,
        f"Overall -- Baseline: PSNR={b_ov['psnr_mean']:.2f}, SSIM={b_ov['ssim_mean']:.4f} | "
        f"Optimized: PSNR={o_ov['psnr_mean']:.2f}, SSIM={o_ov['ssim_mean']:.4f} | "
        f"Delta: PSNR={o_ov['psnr_mean']-b_ov['psnr_mean']:+.2f}, SSIM={o_ov['ssim_mean']-b_ov['ssim_mean']:+.4f}",
        ha="center", fontsize=10, fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Comparison chart saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare Baseline vs Optimized results")
    parser.add_argument("--baseline_results", type=str, default="./results/baseline/metrics.json")
    parser.add_argument("--optimized_results", type=str, default="./results/optimized/metrics.json")
    parser.add_argument("--output_dir", type=str, default="./results/comparison")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_results(args.baseline_results)
    optimized = load_results(args.optimized_results)

    # Print comparison table
    print_comparison_table(baseline, optimized)

    # Create comparison chart
    create_comparison_chart(baseline, optimized, output_dir / "comparison_chart.png")

    # Save combined results
    combined = {
        "baseline": baseline,
        "optimized": optimized,
        "improvement": {
            "psnr_delta": optimized["overall"]["psnr_mean"] - baseline["overall"]["psnr_mean"],
            "ssim_delta": optimized["overall"]["ssim_mean"] - baseline["overall"]["ssim_mean"],
        },
    }
    with open(output_dir / "combined_results.json", "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\nAll comparison artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
