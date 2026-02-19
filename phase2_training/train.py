#!/usr/bin/env python3
"""
MM-GAN Training Script for IXI Dataset
Adapted from: https://github.com/trane293/mm-gan

Supports:
  - Resumable training (checkpoint save/load for 2h Kaggle sessions)
  - Baseline vs Optimized (N4) comparison
  - Curriculum learning for missing modality scenarios
  - LSGAN loss + L1 reconstruction
  - Implicit conditioning

Usage:
  python train.py --data_dir ../data/processed/baseline --experiment baseline
  python train.py --data_dir ../data/processed/optimized_n4 --experiment optimized
  python train.py --resume --experiment baseline  # Resume training
"""

import os
import sys
import argparse
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models.mmgan import (
    GeneratorUNet, Discriminator, weights_init_normal, set_seed,
    ALL_SCENARIOS_3MOD, MODALITY_NAMES,
    get_curriculum_scenarios, impute_missing, impute_reals_into_fake,
    compute_missing_loss,
)
from data.dataset import create_dataloaders
from utils.metrics import psnr_torch, ssim_torch
from utils.checkpoint import (
    save_checkpoint, load_checkpoint, resume_training,
    build_checkpoint_state,
)


def parse_args():
    parser = argparse.ArgumentParser(description="MM-GAN Training for IXI")

    # Data
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to processed slice directory (baseline or optimized_n4)")
    parser.add_argument("--experiment", type=str, default="baseline",
                        help="Experiment name (baseline or optimized)")

    # Training hyperparameters
    parser.add_argument("--n_epochs", type=int, default=60,
                        help="Total number of epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--lr_g", type=float, default=2e-4,
                        help="Generator learning rate")
    parser.add_argument("--lr_d", type=float, default=2e-4,
                        help="Discriminator learning rate")
    parser.add_argument("--lambda_pixel", type=float, default=0.9,
                        help="Weight for L1 pixel loss (1-lambda for GAN loss)")
    parser.add_argument("--beta1", type=float, default=0.5,
                        help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="Adam beta2")

    # Model
    parser.add_argument("--in_channels", type=int, default=3,
                        help="Number of input modalities (T1, T2, PD)")
    parser.add_argument("--depth", type=int, default=6, choices=[6, 8],
                        help="UNet depth: 6 (default, matches paper) or 8 (deeper)")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Image size (square)")

    # Missing modality
    parser.add_argument("--impute_type", type=str, default="zeros",
                        choices=["zeros", "noise", "average"],
                        help="How to impute missing modalities")
    parser.add_argument("--implicit_conditioning", action="store_true", default=True,
                        help="Use implicit conditioning (copy real back)")
    parser.add_argument("--curriculum_learning", action="store_true", default=True,
                        help="Use curriculum learning for scenarios")

    # Checkpoints and logging
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Directory for saving checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="TensorBoard log directory")
    parser.add_argument("--results_dir", type=str, default="./results",
                        help="Directory for saving results")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint")
    parser.add_argument("--save_interval", type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--val_interval", type=int, default=2,
                        help="Validate every N epochs")

    # System
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader workers")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto', 'cuda', or 'cpu'")

    return parser.parse_args()


def validate(generator, val_loader, device, scenarios=None):
    """
    Run validation and compute PSNR/SSIM metrics.

    Tests all missing modality scenarios on the validation set.
    """
    generator.eval()

    if scenarios is None:
        scenarios = ALL_SCENARIOS_3MOD

    all_psnr = []
    all_ssim = []

    with torch.no_grad():
        for batch in val_loader:
            x_real = batch["image"].to(device)

            for scenario in scenarios:
                # Create imputed input
                x_input = impute_missing(x_real, scenario, impute_type="zeros")

                # Generate
                fake_x = generator(x_input)

                # Apply implicit conditioning
                fake_x = impute_reals_into_fake(x_real, fake_x, scenario)

                # Compute metrics only on missing channels
                for idx, available in enumerate(scenario):
                    if available == 0:
                        pred = fake_x[:, idx:idx+1, ...]
                        gt = x_real[:, idx:idx+1, ...]

                        psnr_val = psnr_torch(pred, gt, data_range=1.0).item()
                        ssim_val = ssim_torch(pred, gt, data_range=1.0).item()

                        all_psnr.append(psnr_val)
                        all_ssim.append(ssim_val)

    generator.train()

    avg_psnr = np.mean(all_psnr) if all_psnr else 0.0
    avg_ssim = np.mean(all_ssim) if all_ssim else 0.0

    return avg_psnr, avg_ssim


def train_one_epoch(
    generator, discriminator,
    optimizer_G, optimizer_D,
    train_loader, device,
    criterion_GAN, criterion_pixel,
    lambda_pixel, epoch, n_epochs,
    impute_type="zeros",
    use_ic=True,
    use_curriculum=True,
    writer=None,
    global_step=0,
):
    """Train for one epoch."""
    generator.train()
    discriminator.train()

    epoch_loss_G = 0.0
    epoch_loss_D = 0.0
    n_batches = 0

    # Get scenario range for curriculum learning
    if use_curriculum:
        low, high = get_curriculum_scenarios(epoch, n_epochs)
        available_scenarios = ALL_SCENARIOS_3MOD[low:high]
    else:
        available_scenarios = ALL_SCENARIOS_3MOD

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}", leave=False):
        x_real = batch["image"].to(device)
        B = x_real.size(0)

        # Randomly pick a scenario
        scenario = random.choice(available_scenarios)

        # Create imputed input
        x_input = impute_missing(x_real, scenario, impute_type=impute_type)

        # ------- Train Generator -------
        optimizer_G.zero_grad()

        fake_x = generator(x_input)

        # Implicit conditioning: copy real channels back
        if use_ic:
            fake_x_ic = impute_reals_into_fake(x_real, fake_x, scenario)
        else:
            fake_x_ic = fake_x

        # GAN loss: D should think fake is real
        pred_fake = discriminator(fake_x_ic, x_real)

        # Build per-channel labels for discriminator
        # For GAN loss, we want D to output 1 for all channels
        valid = torch.ones_like(pred_fake, device=device)
        loss_GAN = criterion_GAN(pred_fake, valid)

        # Pixel-wise L1 loss (only on missing channels if IC)
        if use_ic:
            loss_pixel = compute_missing_loss(fake_x_ic, x_real, scenario, criterion_pixel)
        else:
            loss_pixel = criterion_pixel(fake_x, x_real)

        # Total generator loss
        loss_G = (1 - lambda_pixel) * loss_GAN + lambda_pixel * loss_pixel
        loss_G.backward()
        optimizer_G.step()

        # ------- Train Discriminator -------
        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator(x_real, x_real)
        valid = torch.ones_like(pred_real, device=device)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = discriminator(fake_x_ic.detach(), x_real)
        fake_label = torch.zeros_like(pred_fake, device=device)
        loss_fake = criterion_GAN(pred_fake, fake_label)

        # Total discriminator loss
        loss_D = 0.5 * (loss_real + loss_fake)
        loss_D.backward()
        optimizer_D.step()

        epoch_loss_G += loss_G.item()
        epoch_loss_D += loss_D.item()
        n_batches += 1
        global_step += 1

        # TensorBoard logging
        if writer and n_batches % 50 == 0:
            writer.add_scalar("train/loss_G", loss_G.item(), global_step)
            writer.add_scalar("train/loss_D", loss_D.item(), global_step)
            writer.add_scalar("train/loss_GAN", loss_GAN.item(), global_step)
            writer.add_scalar("train/loss_pixel", loss_pixel.item(), global_step)

    avg_loss_G = epoch_loss_G / max(n_batches, 1)
    avg_loss_D = epoch_loss_D / max(n_batches, 1)

    return avg_loss_G, avg_loss_D, global_step


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Experiment directories
    exp_name = args.experiment
    checkpoint_dir = Path(args.checkpoint_dir) / exp_name
    log_dir = Path(args.log_dir) / exp_name
    results_dir = Path(args.results_dir) / exp_name
    for d in [checkpoint_dir, log_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Data
    print("Loading data...")
    loaders = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        target_size=(args.img_size, args.img_size),
        num_workers=args.num_workers,
        augment_train=True,
    )

    if "train" not in loaders:
        print("ERROR: No training data found!")
        sys.exit(1)

    train_loader = loaders["train"]
    val_loader = loaders.get("val", None)

    print(f"Train batches: {len(train_loader)}")
    if val_loader:
        print(f"Val batches:   {len(val_loader)}")

    # Models
    print("Building models...")
    generator = GeneratorUNet(
        in_channels=args.in_channels,
        out_channels=args.in_channels,
        depth=args.depth,
    ).to(device)

    discriminator = Discriminator(
        in_channels=args.in_channels,
        out_channels=args.in_channels,
    ).to(device)

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2)
    )
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2)
    )

    # Schedulers (step LR decay)
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=20, gamma=0.5)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=20, gamma=0.5)

    # Loss functions
    criterion_GAN = nn.MSELoss().to(device)      # LSGAN
    criterion_pixel = nn.L1Loss().to(device)      # L1 reconstruction

    # Resume from checkpoint
    start_epoch = 0
    best_psnr = 0.0
    best_ssim = 0.0
    history = {"train_loss_G": [], "train_loss_D": [], "val_psnr": [], "val_ssim": []}

    if args.resume:
        start_epoch, best_psnr, best_ssim, history = resume_training(
            generator, discriminator,
            optimizer_G, optimizer_D,
            str(checkpoint_dir),
            scheduler_G=scheduler_G,
            scheduler_D=scheduler_D,
        )

    # TensorBoard
    writer = SummaryWriter(str(log_dir))

    # Training loop
    print("=" * 60)
    print(f"Experiment: {exp_name}")
    print(f"Epochs: {start_epoch} -> {args.n_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Lambda pixel: {args.lambda_pixel}")
    print(f"Impute type: {args.impute_type}")
    print(f"IC: {args.implicit_conditioning}")
    print(f"Curriculum: {args.curriculum_learning}")
    print("=" * 60)

    global_step = start_epoch * len(train_loader)
    training_start = time.time()

    for epoch in range(start_epoch, args.n_epochs):
        epoch_start = time.time()

        # Train
        avg_loss_G, avg_loss_D, global_step = train_one_epoch(
            generator, discriminator,
            optimizer_G, optimizer_D,
            train_loader, device,
            criterion_GAN, criterion_pixel,
            args.lambda_pixel, epoch, args.n_epochs,
            impute_type=args.impute_type,
            use_ic=args.implicit_conditioning,
            use_curriculum=args.curriculum_learning,
            writer=writer,
            global_step=global_step,
        )

        # Update schedulers
        scheduler_G.step()
        scheduler_D.step()

        history["train_loss_G"].append(avg_loss_G)
        history["train_loss_D"].append(avg_loss_D)

        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch+1}/{args.n_epochs}] "
              f"Loss_G: {avg_loss_G:.4f} | Loss_D: {avg_loss_D:.4f} | "
              f"Time: {epoch_time:.1f}s")

        # Validation
        if val_loader and (epoch + 1) % args.val_interval == 0:
            avg_psnr, avg_ssim = validate(generator, val_loader, device)
            history["val_psnr"].append(avg_psnr)
            history["val_ssim"].append(avg_ssim)

            writer.add_scalar("val/psnr", avg_psnr, epoch)
            writer.add_scalar("val/ssim", avg_ssim, epoch)

            print(f"  Val PSNR: {avg_psnr:.4f} | Val SSIM: {avg_ssim:.4f}")

            is_best = avg_psnr > best_psnr
            if is_best:
                best_psnr = avg_psnr
                best_ssim = avg_ssim
        else:
            is_best = False

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0 or is_best or (epoch + 1) == args.n_epochs:
            state = build_checkpoint_state(
                epoch, generator, discriminator,
                optimizer_G, optimizer_D,
                best_psnr, best_ssim, history,
                scheduler_G, scheduler_D,
            )
            save_checkpoint(state, str(checkpoint_dir), epoch, is_best=is_best)

    total_time = time.time() - training_start
    writer.close()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"  Total time: {total_time:.1f}s ({total_time/3600:.2f}h)")
    print(f"  Best PSNR:  {best_psnr:.4f}")
    print(f"  Best SSIM:  {best_ssim:.4f}")
    print(f"  Checkpoints: {checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
