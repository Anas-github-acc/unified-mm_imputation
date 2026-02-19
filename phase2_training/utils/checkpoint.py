"""
Checkpoint management utilities for resumable training.

Designed for 2-hour Kaggle session chunks:
  - Saves full training state (model, optimizer, scheduler, epoch, metrics, RNG)
  - Supports seamless resume from latest or specific checkpoint
  - Maintains best-model tracking
"""

import os
import json
import glob
from pathlib import Path
from datetime import datetime

import torch
import numpy as np


def save_checkpoint(
    state,
    checkpoint_dir,
    epoch,
    is_best=False,
    prefix="mmgan",
    max_keep=3,
):
    """
    Save training checkpoint with metadata.

    Args:
        state: dict containing:
            - epoch, generator_state_dict, discriminator_state_dict,
            - optimizer_G_state_dict, optimizer_D_state_dict,
            - scheduler_G_state_dict (optional), scheduler_D_state_dict (optional),
            - best_psnr, best_ssim, train_losses, val_metrics,
            - rng_state, cuda_rng_state, numpy_rng_state
        checkpoint_dir: Directory to save checkpoints
        epoch: Current epoch number
        is_best: Whether this is the best model so far
        prefix: Filename prefix
        max_keep: Maximum number of regular checkpoints to keep
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save epoch checkpoint
    filename = ckpt_dir / f"{prefix}_epoch_{epoch:04d}.pth"
    torch.save(state, filename)
    print(f"  [CKPT] Saved: {filename}")

    # Save latest pointer
    latest_path = ckpt_dir / f"{prefix}_latest.pth"
    torch.save(state, latest_path)

    # Save best model
    if is_best:
        best_path = ckpt_dir / f"{prefix}_best.pth"
        torch.save(state, best_path)
        print(f"  [CKPT] New best model saved!")

    # Save metadata
    meta = {
        "epoch": epoch,
        "timestamp": datetime.now().isoformat(),
        "is_best": is_best,
        "best_psnr": state.get("best_psnr", 0.0),
        "best_ssim": state.get("best_ssim", 0.0),
    }
    meta_path = ckpt_dir / f"{prefix}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Cleanup old checkpoints (keep latest N)
    all_ckpts = sorted(glob.glob(str(ckpt_dir / f"{prefix}_epoch_*.pth")))
    if len(all_ckpts) > max_keep:
        for old_ckpt in all_ckpts[:-max_keep]:
            os.remove(old_ckpt)


def load_checkpoint(checkpoint_dir, prefix="mmgan", which="latest"):
    """
    Load a checkpoint.

    Args:
        checkpoint_dir: Directory containing checkpoints
        prefix: Filename prefix
        which: 'latest', 'best', or epoch number (int)

    Returns:
        state dict or None if not found
    """
    ckpt_dir = Path(checkpoint_dir)

    if isinstance(which, int):
        path = ckpt_dir / f"{prefix}_epoch_{which:04d}.pth"
    elif which == "best":
        path = ckpt_dir / f"{prefix}_best.pth"
    else:
        path = ckpt_dir / f"{prefix}_latest.pth"

    if not path.exists():
        print(f"  [CKPT] No checkpoint found at: {path}")
        return None

    print(f"  [CKPT] Loading: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    return state


def resume_training(
    generator, discriminator,
    optimizer_G, optimizer_D,
    checkpoint_dir, prefix="mmgan",
    scheduler_G=None, scheduler_D=None,
):
    """
    Resume training from latest checkpoint.

    Returns:
        start_epoch: Epoch to resume from (0 if no checkpoint)
        best_psnr: Best PSNR so far
        best_ssim: Best SSIM so far
        history: Training history dict
    """
    state = load_checkpoint(checkpoint_dir, prefix, which="latest")

    if state is None:
        return 0, 0.0, 0.0, {"train_loss_G": [], "train_loss_D": [], "val_psnr": [], "val_ssim": []}

    # Restore model states
    generator.load_state_dict(state["generator_state_dict"])
    discriminator.load_state_dict(state["discriminator_state_dict"])
    optimizer_G.load_state_dict(state["optimizer_G_state_dict"])
    optimizer_D.load_state_dict(state["optimizer_D_state_dict"])

    if scheduler_G is not None and "scheduler_G_state_dict" in state:
        scheduler_G.load_state_dict(state["scheduler_G_state_dict"])
    if scheduler_D is not None and "scheduler_D_state_dict" in state:
        scheduler_D.load_state_dict(state["scheduler_D_state_dict"])

    # Restore RNG states for reproducibility
    if "rng_state" in state:
        torch.set_rng_state(state["rng_state"])
    if "numpy_rng_state" in state:
        np.random.set_state(state["numpy_rng_state"])

    start_epoch = state["epoch"] + 1
    best_psnr = state.get("best_psnr", 0.0)
    best_ssim = state.get("best_ssim", 0.0)

    history = state.get("history", {
        "train_loss_G": [], "train_loss_D": [],
        "val_psnr": [], "val_ssim": [],
    })

    print(f"  [CKPT] Resuming from epoch {start_epoch}")
    print(f"  [CKPT] Best PSNR: {best_psnr:.4f}, Best SSIM: {best_ssim:.4f}")

    return start_epoch, best_psnr, best_ssim, history


def build_checkpoint_state(
    epoch, generator, discriminator,
    optimizer_G, optimizer_D,
    best_psnr, best_ssim, history,
    scheduler_G=None, scheduler_D=None,
):
    """Build a checkpoint state dict."""
    state = {
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_G_state_dict": optimizer_G.state_dict(),
        "optimizer_D_state_dict": optimizer_D.state_dict(),
        "best_psnr": best_psnr,
        "best_ssim": best_ssim,
        "history": history,
        "rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
    }

    if scheduler_G is not None:
        state["scheduler_G_state_dict"] = scheduler_G.state_dict()
    if scheduler_D is not None:
        state["scheduler_D_state_dict"] = scheduler_D.state_dict()

    return state
