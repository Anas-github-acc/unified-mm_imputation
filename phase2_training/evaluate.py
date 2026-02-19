#!/usr/bin/env python3
"""
Evaluation Script for MM-GAN on IXI Dataset

Computes PSNR and SSIM metrics on test set for all missing modality scenarios.
Generates comparison figures and result tables.

Usage:
  python evaluate.py --data_dir ../data/processed/baseline \
                     --checkpoint ./checkpoints/baseline/mmgan_best.pth \
                     --experiment baseline

  python evaluate.py --data_dir ../data/processed/optimized_n4 \
                     --checkpoint ./checkpoints/optimized/mmgan_best.pth \
                     --experiment optimized
"""

import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from models.mmgan import (
    GeneratorUNet, ALL_SCENARIOS_3MOD, MODALITY_NAMES,
    impute_missing, impute_reals_into_fake,
)
from data.dataset import create_dataloaders
from utils.metrics import psnr_numpy, ssim_numpy
from utils.checkpoint import load_checkpoint


def evaluate_model(
    generator, test_loader, device,
    scenarios=None, impute_type="zeros",
):
    """
    Full evaluation on test set.

    Returns:
        results: dict keyed by scenario string, each containing
                 lists of per-sample PSNR and SSIM values
        overall: dict with mean PSNR and SSIM across all scenarios
    """
    generator.eval()

    if scenarios is None:
        scenarios = ALL_SCENARIOS_3MOD

    results = {}

    with torch.no_grad():
        for scenario in scenarios:
            scenario_str = "".join(str(s) for s in scenario)
            missing_mods = [MODALITY_NAMES[i] for i, s in enumerate(scenario) if s == 0]
            avail_mods = [MODALITY_NAMES[i] for i, s in enumerate(scenario) if s == 1]

            psnr_list = []
            ssim_list = []

            for batch in tqdm(test_loader, desc=f"Scenario {scenario_str}", leave=False):
                x_real = batch["image"].to(device)
                B = x_real.size(0)

                # Impute missing
                x_input = impute_missing(x_real, scenario, impute_type=impute_type)

                # Generate
                fake_x = generator(x_input)

                # Implicit conditioning
                fake_x = impute_reals_into_fake(x_real, fake_x, scenario)

                # Move to numpy for metrics
                fake_np = fake_x.cpu().numpy()
                real_np = x_real.cpu().numpy()

                # Compute metrics on missing channels
                for b in range(B):
                    for idx, available in enumerate(scenario):
                        if available == 0:
                            pred = fake_np[b, idx]
                            gt = real_np[b, idx]

                            psnr_val = psnr_numpy(pred, gt, data_range=1.0)
                            ssim_val = ssim_numpy(pred, gt, data_range=1.0)

                            psnr_list.append(psnr_val)
                            ssim_list.append(ssim_val)

            results[scenario_str] = {
                "scenario": scenario,
                "missing": missing_mods,
                "available": avail_mods,
                "psnr_mean": float(np.mean(psnr_list)),
                "psnr_std": float(np.std(psnr_list)),
                "ssim_mean": float(np.mean(ssim_list)),
                "ssim_std": float(np.std(ssim_list)),
                "n_samples": len(psnr_list),
            }

    # Overall metrics
    all_psnr = []
    all_ssim = []
    for v in results.values():
        all_psnr.append(v["psnr_mean"])
        all_ssim.append(v["ssim_mean"])

    overall = {
        "psnr_mean": float(np.mean(all_psnr)),
        "ssim_mean": float(np.mean(all_ssim)),
    }

    return results, overall


def generate_visual_comparison(
    generator, test_loader, device, output_dir, n_samples=5,
):
    """
    Generate side-by-side visual comparison images.
    Shows: Input modalities | Ground truth | Synthesized output
    """
    generator.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios_to_show = [
        [0, 1, 1],  # Missing T1
        [1, 0, 1],  # Missing T2
        [1, 1, 0],  # Missing PD
        [1, 0, 0],  # Only T1
    ]

    with torch.no_grad():
        # Get a batch
        batch = next(iter(test_loader))
        x_real = batch["image"][:n_samples].to(device)

        for scenario in scenarios_to_show:
            scenario_str = "".join(str(s) for s in scenario)
            missing_mods = [MODALITY_NAMES[i] for i, s in enumerate(scenario) if s == 0]
            avail_mods = [MODALITY_NAMES[i] for i, s in enumerate(scenario) if s == 1]

            x_input = impute_missing(x_real, scenario, impute_type="zeros")
            fake_x = generator(x_input)
            fake_x = impute_reals_into_fake(x_real, fake_x, scenario)

            # Create figure
            n_cols = 3 + len(missing_mods)  # Input mods + GT + Synth for each missing
            fig, axes = plt.subplots(n_samples, n_cols, figsize=(4 * n_cols, 4 * n_samples))

            if n_samples == 1:
                axes = axes[np.newaxis, :]

            for sample_idx in range(n_samples):
                col = 0

                # Show available modalities (input)
                for mod_idx, mod_name in enumerate(MODALITY_NAMES):
                    if scenario[mod_idx] == 1:
                        axes[sample_idx, col].imshow(
                            x_real[sample_idx, mod_idx].cpu().numpy(),
                            cmap="gray", vmin=0, vmax=1,
                        )
                        axes[sample_idx, col].set_title(f"Input: {mod_name}")
                        axes[sample_idx, col].axis("off")
                        col += 1

                # Show ground truth and synthesized for missing modalities
                for mod_idx, mod_name in enumerate(MODALITY_NAMES):
                    if scenario[mod_idx] == 0:
                        # Ground truth
                        axes[sample_idx, col].imshow(
                            x_real[sample_idx, mod_idx].cpu().numpy(),
                            cmap="gray", vmin=0, vmax=1,
                        )
                        axes[sample_idx, col].set_title(f"GT: {mod_name}")
                        axes[sample_idx, col].axis("off")
                        col += 1

                        # Synthesized
                        axes[sample_idx, col].imshow(
                            fake_x[sample_idx, mod_idx].cpu().numpy(),
                            cmap="gray", vmin=0, vmax=1,
                        )
                        axes[sample_idx, col].set_title(f"Synth: {mod_name}")
                        axes[sample_idx, col].axis("off")
                        col += 1

                # Fill remaining columns
                while col < n_cols:
                    axes[sample_idx, col].axis("off")
                    col += 1

            fig.suptitle(
                f"Scenario: Available={avail_mods}, Missing={missing_mods}",
                fontsize=14, fontweight="bold",
            )
            plt.tight_layout()
            plt.savefig(output_dir / f"comparison_{scenario_str}.png", dpi=150, bbox_inches="tight")
            plt.close()

    print(f"Visual comparisons saved to: {output_dir}")


def print_results_table(results, overall, experiment_name):
    """Print a formatted results table."""
    print(f"\n{'='*70}")
    print(f"Results: {experiment_name}")
    print(f"{'='*70}")
    print(f"{'Scenario':<12} {'Missing':<15} {'Available':<15} {'PSNR':>10} {'SSIM':>10}")
    print(f"{'-'*70}")

    for key, val in results.items():
        missing = ",".join(val["missing"])
        avail = ",".join(val["available"])
        print(f"{key:<12} {missing:<15} {avail:<15} "
              f"{val['psnr_mean']:>7.2f}+-{val['psnr_std']:.2f} "
              f"{val['ssim_mean']:>7.4f}+-{val['ssim_std']:.4f}")

    print(f"{'-'*70}")
    print(f"{'OVERALL':<42} {overall['psnr_mean']:>7.2f}       {overall['ssim_mean']:>7.4f}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="MM-GAN Evaluation")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file or directory")
    parser.add_argument("--experiment", type=str, default="baseline")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--n_visual_samples", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Output directory
    results_dir = Path(args.output_dir) / args.experiment
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model...")
    generator = GeneratorUNet(
        in_channels=args.in_channels,
        out_channels=args.in_channels,
    ).to(device)

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.is_dir():
        state = load_checkpoint(str(ckpt_path))
    else:
        state = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    if state is None:
        print("ERROR: Could not load checkpoint!")
        sys.exit(1)

    generator.load_state_dict(state["generator_state_dict"])
    print(f"Loaded checkpoint (epoch {state.get('epoch', '?')})")

    # Load test data
    print("Loading test data...")
    loaders = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        target_size=(args.img_size, args.img_size),
        num_workers=0,
        augment_train=False,
    )

    if "test" not in loaders:
        print("ERROR: No test split found!")
        sys.exit(1)

    test_loader = loaders["test"]
    print(f"Test batches: {len(test_loader)}")

    # Evaluate
    print("\nRunning evaluation...")
    results, overall = evaluate_model(generator, test_loader, device)

    # Print table
    print_results_table(results, overall, args.experiment)

    # Save results
    results_path = results_dir / "metrics.json"
    with open(results_path, "w") as f:
        json.dump({"per_scenario": results, "overall": overall}, f, indent=2)
    print(f"\nMetrics saved to: {results_path}")

    # Generate visual comparisons
    print("\nGenerating visual comparisons...")
    generate_visual_comparison(
        generator, test_loader, device,
        output_dir=results_dir / "visuals",
        n_samples=args.n_visual_samples,
    )

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
