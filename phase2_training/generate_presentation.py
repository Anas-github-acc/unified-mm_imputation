#!/usr/bin/env python3
"""
Generate Presentation Figures for MM-GAN Missing Modality Imputation

Generates all figures needed for a presentation:
  1. Side-by-side visual comparisons (Input | GT | Synthesized | Error) per scenario
  2. Full 6-scenario grid per test subject
  3. PSNR bar chart with paper's Table II reference targets
  4. SSIM bar chart with paper's Table II reference targets
  5. Training loss curves (from checkpoint history)
  6. Validation metric curves with paper reference lines
  7. Formatted results table (console + saved to text file)

Three data modes:
  --demo       Generate synthetic brain phantoms (instant, no download)
  --ixi_dir    Load raw IXI NIfTI files (after downloading tar archives)
  --data_dir   Use pre-extracted .npy slices from Phase 1

Usage:
  # With trained model + real test data (RECOMMENDED):
  python generate_presentation.py \\
      --data_dir ../data/processed/baseline \\
      --checkpoint ./comparer/baseline/checkpoints/baseline \\
      --output_dir ./presentation

  # Quick mode (5 subjects for visuals, 200 slices for metrics):
  python generate_presentation.py \\
      --data_dir ../data/processed/baseline \\
      --checkpoint ./comparer/baseline/checkpoints/baseline \\
      --n_visual 5 --n_eval 200 --output_dir ./presentation

  # Full evaluation on all test slices:
  python generate_presentation.py \\
      --data_dir ../data/processed/baseline \\
      --checkpoint ./comparer/baseline/checkpoints/baseline \\
      --n_eval 0 --output_dir ./presentation_full

  # Use pre-computed metrics (skip inference, just make figures):
  python generate_presentation.py \\
      --data_dir ../data/processed/baseline \\
      --checkpoint ./comparer/baseline/checkpoints/baseline \\
      --metrics_json ./comparer/baseline/results/baseline/metrics.json \\
      --output_dir ./presentation

  # Demo mode (synthetic phantoms, no data needed):
  python generate_presentation.py --demo --output_dir ./presentation_demo
"""

import os
import sys
import re
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent))

from models.mmgan import (
    GeneratorUNet, weights_init_normal, set_seed,
    ALL_SCENARIOS_3MOD, MODALITY_NAMES,
    impute_missing, impute_reals_into_fake,
)
from utils.metrics import psnr_numpy, ssim_numpy
from utils.checkpoint import load_checkpoint


# ============================================================
# Paper's Table II IXI benchmark targets
# ============================================================
PAPER_TARGETS = {
    "110": {  # Available: T1, T2 -> Missing: PD
        "psnr_mean": 31.86, "psnr_std": 0.41,
        "ssim_mean": 0.980, "ssim_std": 0.004,
        "label": "T1,T2 -> PD",
    },
    "011": {  # Available: T2, PD -> Missing: T1
        "psnr_mean": 31.32, "psnr_std": 0.49,
        "ssim_mean": 0.948, "ssim_std": 0.025,
        "label": "T2,PD -> T1",
    },
}

SCENARIO_LABELS = {
    "100": "Only T1\n(miss T2,PD)",
    "010": "Only T2\n(miss T1,PD)",
    "001": "Only PD\n(miss T1,T2)",
    "110": "T1+T2\n(miss PD)",
    "101": "T1+PD\n(miss T2)",
    "011": "T2+PD\n(miss T1)",
}

SCENARIO_LABELS_ONELINE = {
    "100": "T1 -> T2,PD",
    "010": "T2 -> T1,PD",
    "001": "PD -> T1,T2",
    "110": "T1,T2 -> PD",
    "101": "T1,PD -> T2",
    "011": "T2,PD -> T1",
}

# Model expects 256x256
MODEL_SIZE = (256, 256)


# ============================================================
# Demo Mode: Synthetic brain-like phantom data
# ============================================================

def _make_ellipse_mask(shape, center, radii, rotation_deg=0):
    """Create a 2-D binary ellipse mask."""
    h, w = shape
    y, x = np.ogrid[:h, :w]
    cy, cx = center
    dy = y - cy
    dx = x - cx

    if rotation_deg != 0:
        theta = np.deg2rad(rotation_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        dx_r = cos_t * dx + sin_t * dy
        dy_r = -sin_t * dx + cos_t * dy
    else:
        dx_r, dy_r = dx, dy

    ry, rx = radii
    return ((dx_r / rx) ** 2 + (dy_r / ry) ** 2) <= 1.0


def _generate_phantom_slice(size=256, seed=0):
    """Generate a synthetic brain-like axial slice (3, size, size) in [0,1]."""
    rng = np.random.RandomState(seed)
    h = w = size
    c = size // 2

    t1 = np.zeros((h, w), dtype=np.float32)
    t2 = np.zeros((h, w), dtype=np.float32)
    pd = np.zeros((h, w), dtype=np.float32)

    skull = _make_ellipse_mask((h, w), (c, c), (c * 0.90, c * 0.82))
    t1 += skull * 0.15; t2 += skull * 0.10; pd += skull * 0.12

    wm = _make_ellipse_mask((h, w), (c, c), (c * 0.80, c * 0.72))
    t1[wm] = 0.70; t2[wm] = 0.35; pd[wm] = 0.65

    gm_outer = _make_ellipse_mask((h, w), (c, c), (c * 0.82, c * 0.74))
    gm_inner = _make_ellipse_mask((h, w), (c, c), (c * 0.72, c * 0.64))
    gm = gm_outer & ~gm_inner
    t1[gm] = 0.50; t2[gm] = 0.55; pd[gm] = 0.80

    for dx_off, dy_off in [(-0.18, 0.0), (0.18, 0.0)]:
        dgm = _make_ellipse_mask(
            (h, w), (c + int(dy_off * c), c + int(dx_off * c)),
            (c * 0.12, c * 0.08))
        t1[dgm] = 0.45; t2[dgm] = 0.50; pd[dgm] = 0.78

    for dx_off in [-0.07, 0.07]:
        vent = _make_ellipse_mask(
            (h, w), (c - int(0.02 * c), c + int(dx_off * c)),
            (c * 0.18, c * 0.04), rotation_deg=rng.uniform(-5, 5))
        t1[vent] = 0.10; t2[vent] = 0.95; pd[vent] = 0.95

    for ch in [t1, t2, pd]:
        ch += rng.normal(0, 0.02, ch.shape).astype(np.float32)

    try:
        from scipy.ndimage import gaussian_filter
        t1 = gaussian_filter(t1, sigma=1.2)
        t2 = gaussian_filter(t2, sigma=1.2)
        pd = gaussian_filter(pd, sigma=1.2)
    except ImportError:
        pass  # scipy optional for demo

    return np.stack([np.clip(t1, 0, 1), np.clip(t2, 0, 1), np.clip(pd, 0, 1)], axis=0)


def generate_demo_subjects(n_subjects=3):
    """Generate synthetic brain-like phantom slices for demo mode."""
    print("  Generating synthetic brain phantoms (no download needed)...")
    slices = []
    for i in range(n_subjects):
        name = f"Phantom_{i+1:03d}"
        data = _generate_phantom_slice(size=MODEL_SIZE[0], seed=42 + i * 7)
        slices.append((name, data))
        print(f"    {name}: shape {data.shape}")
    return slices


# ============================================================
# Data Loading: .npy slices (from Phase 1 preprocessing)
# ============================================================

def _group_slices_by_subject(npy_files):
    """Group .npy filenames by IXI subject ID.
    Filename format: IXI{ID}_slice{NUM}.npy
    """
    pat = re.compile(r"(IXI\d+)_slice(\d+)")
    groups = defaultdict(list)
    for fp in npy_files:
        m = pat.match(fp.stem)
        if m:
            groups[m.group(1)].append((int(m.group(2)), fp))
        else:
            groups[fp.stem].append((0, fp))
    # Sort each subject's slices by slice number
    for k in groups:
        groups[k].sort(key=lambda x: x[0])
    return dict(groups)


def load_npy_slices_for_visual(data_dir, n_subjects=5):
    """
    Load one representative slice per subject for visual figures.
    Picks the middle slice (best brain content) for each subject.

    Returns:
        list of (subject_name, np.ndarray (3, H, W))
    """
    data_dir = Path(data_dir)

    # Find the test split directory
    for candidate in [data_dir / "test", data_dir]:
        npy_files = sorted(candidate.glob("*.npy"))
        if npy_files:
            data_dir = candidate
            break

    if not npy_files:
        print(f"ERROR: No .npy files found in {data_dir}")
        sys.exit(1)

    groups = _group_slices_by_subject(npy_files)
    subject_ids = sorted(groups.keys())

    # Pick evenly spaced subjects
    n = min(n_subjects, len(subject_ids))
    indices = np.linspace(0, len(subject_ids) - 1, n, dtype=int)

    slices = []
    for idx in indices:
        subj_id = subject_ids[idx]
        subj_slices = groups[subj_id]
        # Pick middle slice
        mid = len(subj_slices) // 2
        _, fp = subj_slices[mid]
        data = np.load(fp).astype(np.float32)
        slices.append((subj_id, data))
        print(f"  {subj_id}: slice {mid}/{len(subj_slices)}, shape {data.shape}")

    return slices


def load_npy_slices_for_eval(data_dir, n_slices=200):
    """
    Load slices for quantitative evaluation.
    If n_slices=0, load ALL test slices.

    Returns:
        list of (filename_stem, np.ndarray (3, H, W))
    """
    data_dir = Path(data_dir)

    for candidate in [data_dir / "test", data_dir]:
        npy_files = sorted(candidate.glob("*.npy"))
        if npy_files:
            break

    if not npy_files:
        print(f"ERROR: No .npy files found in {data_dir}")
        sys.exit(1)

    if n_slices == 0 or n_slices >= len(npy_files):
        selected = npy_files
        print(f"  Loading ALL {len(selected)} test slices...")
    else:
        indices = np.linspace(0, len(npy_files) - 1, n_slices, dtype=int)
        selected = [npy_files[i] for i in indices]
        print(f"  Loading {len(selected)}/{len(npy_files)} test slices (evenly sampled)...")

    slices = []
    for fp in selected:
        data = np.load(fp).astype(np.float32)
        slices.append((fp.stem, data))

    print(f"  Loaded {len(slices)} slices, shape: {slices[0][1].shape}")
    return slices


def load_raw_nifti_slices(ixi_dir, n_subjects=3):
    """Load raw NIfTI files from the IXI dataset directory."""
    from PIL import Image
    try:
        import nibabel as nib
    except ImportError:
        print("ERROR: nibabel required for --ixi_dir. Install: pip install nibabel")
        sys.exit(1)

    ixi_dir = Path(ixi_dir)
    nii_files = sorted(ixi_dir.glob("IXI*.nii.gz"))
    if not nii_files:
        for mod in ["T1", "T2", "PD"]:
            nii_files.extend(sorted((ixi_dir / mod).glob("IXI*.nii.gz")))

    if not nii_files:
        print(f"ERROR: No IXI NIfTI files found in {ixi_dir}")
        sys.exit(1)

    pat = re.compile(r"(IXI\d+)-\w+-\d+-(T1|T2|PD)")
    subjects = {}
    for fp in nii_files:
        m = pat.match(fp.name)
        if m:
            subjects.setdefault(m.group(1), {})[m.group(2)] = fp

    complete = {k: v for k, v in subjects.items() if len(v) == 3}
    if not complete:
        print(f"ERROR: No subjects with all 3 modalities found")
        sys.exit(1)

    subj_ids = sorted(complete.keys())
    indices = np.linspace(0, len(subj_ids) - 1, min(n_subjects, len(subj_ids)), dtype=int)

    slices = []
    for idx in indices:
        subj_id = subj_ids[idx]
        files = complete[subj_id]
        channels = []
        for mod in ["T1", "T2", "PD"]:
            vol = nib.load(str(files[mod])).get_fdata().astype(np.float32)
            vmin, vmax = vol.min(), vol.max()
            if vmax - vmin > 1e-8:
                vol = (vol - vmin) / (vmax - vmin)
            s = vol[:, :, int(vol.shape[2] * 0.45)]
            h, w = s.shape
            d = min(h, w)
            s = s[(h-d)//2:(h-d)//2+d, (w-d)//2:(w-d)//2+d]
            s = np.array(Image.fromarray(s).resize(MODEL_SIZE, Image.BILINEAR), dtype=np.float32)
            channels.append(s)
        stacked = np.stack(channels, axis=0)
        slices.append((subj_id, stacked))
        print(f"  {subj_id}: shape {stacked.shape}")

    return slices


# ============================================================
# Model loading
# ============================================================

def _detect_depth_from_state(gen_state_dict):
    """Infer UNet depth from checkpoint keys (6 or 8)."""
    if "down7.model.0.weight" in gen_state_dict:
        return 8
    return 6


def load_model(checkpoint_path=None, device="cpu"):
    """Load generator model, optionally from checkpoint.

    Auto-detects the UNet depth (6 or 8) from the checkpoint so that
    checkpoints trained with depth=6 (baseline/optimized) load without
    any architecture mismatch.
    """
    trained = False
    epoch_info = "N/A"
    history = None
    state = None

    if checkpoint_path:
        ckpt_path = Path(checkpoint_path)

        if ckpt_path.is_dir():
            state = load_checkpoint(str(ckpt_path), which="best")
            if state is None:
                state = load_checkpoint(str(ckpt_path), which="latest")
        elif ckpt_path.is_file():
            state = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    if state is not None:
        depth = _detect_depth_from_state(state["generator_state_dict"])
        print(f"  Detected UNet depth={depth} from checkpoint")
        generator = GeneratorUNet(in_channels=3, out_channels=3, depth=depth).to(device)
        generator.load_state_dict(state["generator_state_dict"])
        epoch_info = state.get("epoch", "?")
        history = state.get("history", None)
        trained = True
        best_psnr = state.get("best_psnr", "?")
        best_ssim = state.get("best_ssim", "?")
        print(f"  Loaded trained model (epoch {epoch_info}, "
              f"best_psnr={best_psnr}, best_ssim={best_ssim})")
    else:
        # No checkpoint – default to depth=6 (matches original paper)
        generator = GeneratorUNet(in_channels=3, out_channels=3, depth=6).to(device)
        if checkpoint_path:
            print("  WARNING: Checkpoint not found, using untrained model")
        else:
            print("  No checkpoint provided, using untrained model (random weights)")
        generator.apply(weights_init_normal)

    generator.eval()
    return generator, trained, epoch_info, history


# ============================================================
# Inference + Evaluation
# ============================================================

def run_inference_single(generator, data_np, scenario, device):
    """
    Run inference on a single slice for a single scenario.

    Args:
        generator: model in eval mode
        data_np: np.ndarray (3, H, W)
        scenario: list of 0/1
        device: torch device

    Returns:
        synth_np: np.ndarray (3, H, W) — full synthesized output
    """
    x_real = torch.from_numpy(data_np).unsqueeze(0).to(device)  # (1, 3, H, W)

    # Resize to model size if needed
    _, _, h, w = x_real.shape
    if (h, w) != MODEL_SIZE:
        x_real = torch.nn.functional.interpolate(
            x_real, size=MODEL_SIZE, mode="bilinear", align_corners=False)

    x_input = impute_missing(x_real, scenario, impute_type="zeros")
    fake_x = generator(x_input)
    fake_x = impute_reals_into_fake(x_real, fake_x, scenario)

    # Resize back to original size for fair metric computation
    if (h, w) != MODEL_SIZE:
        fake_x = torch.nn.functional.interpolate(
            fake_x, size=(h, w), mode="bilinear", align_corners=False)
        x_real = torch.nn.functional.interpolate(
            x_real, size=(h, w), mode="bilinear", align_corners=False)

    return fake_x[0].cpu().numpy(), x_real[0].cpu().numpy()


def evaluate_all_scenarios(generator, slices_list, device, verbose=True):
    """
    Run inference on all 6 scenarios for all slices.

    Returns:
        results: dict[scenario_str] -> {psnr_mean, psnr_std, ssim_mean, ssim_std, ...}
        per_sample: dict[scenario_str] -> list of per-sample dicts
        synth_images: dict[(subj_idx, scenario_str)] -> (synth_np, real_np) both (3,H,W)
    """
    generator.eval()
    results = {}
    per_sample = {}
    synth_images = {}

    n_total = len(slices_list)

    with torch.no_grad():
        for scenario in ALL_SCENARIOS_3MOD:
            scenario_str = "".join(str(s) for s in scenario)
            missing_mods = [MODALITY_NAMES[i] for i, s in enumerate(scenario) if s == 0]
            avail_mods = [MODALITY_NAMES[i] for i, s in enumerate(scenario) if s == 1]

            psnr_list = []
            ssim_list = []
            sample_details = []

            for subj_idx, (name, data) in enumerate(slices_list):
                if verbose and (subj_idx + 1) % 50 == 0:
                    print(f"    Scenario {scenario_str}: {subj_idx+1}/{n_total}", flush=True)

                synth_np, real_np = run_inference_single(generator, data, scenario, device)
                synth_images[(subj_idx, scenario_str)] = (synth_np, real_np)

                # Metrics on missing channels only
                for idx, available in enumerate(scenario):
                    if available == 0:
                        pred = np.clip(synth_np[idx], 0, 1)
                        gt = real_np[idx]
                        p = psnr_numpy(pred, gt, data_range=1.0)
                        s = ssim_numpy(pred, gt, data_range=1.0)
                        psnr_list.append(p)
                        ssim_list.append(s)
                        sample_details.append({
                            "subject": name, "modality": MODALITY_NAMES[idx],
                            "psnr": float(p), "ssim": float(s),
                        })

            results[scenario_str] = {
                "scenario": scenario,
                "missing": missing_mods,
                "available": avail_mods,
                "psnr_mean": float(np.mean(psnr_list)) if psnr_list else 0.0,
                "psnr_std": float(np.std(psnr_list)) if psnr_list else 0.0,
                "ssim_mean": float(np.mean(ssim_list)) if ssim_list else 0.0,
                "ssim_std": float(np.std(ssim_list)) if ssim_list else 0.0,
                "n_samples": len(psnr_list),
            }
            per_sample[scenario_str] = sample_details

            if verbose:
                print(f"    {scenario_str} ({SCENARIO_LABELS_ONELINE[scenario_str]}): "
                      f"PSNR={results[scenario_str]['psnr_mean']:.2f} +/- "
                      f"{results[scenario_str]['psnr_std']:.2f}  "
                      f"SSIM={results[scenario_str]['ssim_mean']:.4f} +/- "
                      f"{results[scenario_str]['ssim_std']:.4f}  "
                      f"(n={results[scenario_str]['n_samples']})")

    return results, per_sample, synth_images


# ============================================================
# Figure 1: Side-by-side visual comparisons (with error map)
# ============================================================

def fig_visual_comparison(slices_list, synth_images, output_dir, trained):
    """
    Side-by-side for each single-missing scenario:
    Columns: Available1 | Available2 | GT | Synthesized | Error Map
    Rows: one per subject
    """
    single_missing = [
        ([0, 1, 1], "011", "T1"),
        ([1, 0, 1], "101", "T2"),
        ([1, 1, 0], "110", "PD"),
    ]

    for scenario, scenario_str, missing_mod in single_missing:
        avail_indices = [i for i, s in enumerate(scenario) if s == 1]
        missing_idx = [i for i, s in enumerate(scenario) if s == 0][0]

        n_subjects = len(slices_list)
        fig, axes = plt.subplots(n_subjects, 5, figsize=(20, 4 * n_subjects))
        if n_subjects == 1:
            axes = axes[np.newaxis, :]

        for row, (name, data) in enumerate(slices_list):
            entry = synth_images.get((row, scenario_str))
            if entry is None:
                continue
            synth_np, real_np = entry

            # Use real_np (may have been resized) for display consistency
            for col_idx, (ch_idx, title) in enumerate([
                (avail_indices[0], f"Input: {MODALITY_NAMES[avail_indices[0]]}"),
                (avail_indices[1], f"Input: {MODALITY_NAMES[avail_indices[1]]}"),
            ]):
                ax = axes[row, col_idx]
                ax.imshow(real_np[ch_idx], cmap="gray", vmin=0, vmax=1)
                ax.set_title(title, fontsize=11, fontweight="bold")
                ax.axis("off")

            # GT
            gt = real_np[missing_idx]
            ax = axes[row, 2]
            ax.imshow(gt, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"Ground Truth: {missing_mod}", fontsize=11, fontweight="bold")
            ax.axis("off")

            # Synthesized
            pred = np.clip(synth_np[missing_idx], 0, 1)
            p = psnr_numpy(pred, gt, data_range=1.0)
            s = ssim_numpy(pred, gt, data_range=1.0)
            ax = axes[row, 3]
            ax.imshow(pred, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"Synthesized: {missing_mod}\nPSNR={p:.2f}  SSIM={s:.4f}",
                         fontsize=10, fontweight="bold")
            ax.axis("off")

            # Error map (amplified for visibility)
            error = np.abs(gt - pred)
            ax = axes[row, 4]
            im = ax.imshow(error, cmap="hot", vmin=0, vmax=0.3)
            ax.set_title(f"Error Map (|GT-Synth|)\nMAE={error.mean():.4f}", fontsize=10)
            ax.axis("off")

            # Row label
            axes[row, 0].set_ylabel(name, fontsize=10, rotation=0, labelpad=70, va="center")

        status = "Trained Model (epoch {})".format(trained) if trained else "UNTRAINED"
        fig.suptitle(
            f"Missing {missing_mod} Synthesis  |  Scenario: {SCENARIO_LABELS_ONELINE[scenario_str]}  |  {status}",
            fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        save_path = output_dir / f"visual_missing_{missing_mod}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {save_path}")


# ============================================================
# Figure 2: Full 6-scenario grid per subject
# ============================================================

def fig_subject_grid(slices_list, synth_images, output_dir, trained):
    """
    Per subject: 6 rows (scenarios) x columns showing input/GT/synth.
    """
    for subj_idx, (name, data) in enumerate(slices_list):
        fig = plt.figure(figsize=(32, 26))
        gs = gridspec.GridSpec(6, 5, figure=fig, hspace=0.35, wspace=0.15,
                               width_ratios=[3, 1, 1, 1, 1])

        for row_idx, scenario in enumerate(ALL_SCENARIOS_3MOD):
            scenario_str = "".join(str(s) for s in scenario)
            entry = synth_images.get((subj_idx, scenario_str))
            if entry is None:
                continue
            synth_np, real_np = entry

            H, W = real_np.shape[1], real_np.shape[2]
            missing_mods = [MODALITY_NAMES[i] for i, s in enumerate(scenario) if s == 0]
            avail_mods = [MODALITY_NAMES[i] for i, s in enumerate(scenario) if s == 1]

            # Col 0: Combined input visualization
            ax0 = fig.add_subplot(gs[row_idx, 0])
            gap = 4
            input_vis = np.zeros((H, W * 3 + gap * 2), dtype=np.float32)
            for ch in range(3):
                start = ch * (W + gap)
                if scenario[ch] == 1:
                    input_vis[:, start:start + W] = real_np[ch]
            ax0.imshow(input_vis, cmap="gray", vmin=0, vmax=1, aspect="auto")
            ax0.set_title(f"Input: {'+'.join(avail_mods)}", fontsize=10, fontweight="bold")
            ax0.axis("off")

            # Remaining columns: GT + Synth for each missing modality
            col = 1
            for ch_idx in range(3):
                if scenario[ch_idx] == 0 and col <= 4:
                    mod_name = MODALITY_NAMES[ch_idx]
                    gt = real_np[ch_idx]
                    pred = np.clip(synth_np[ch_idx], 0, 1)

                    ax_gt = fig.add_subplot(gs[row_idx, col])
                    ax_gt.imshow(gt, cmap="gray", vmin=0, vmax=1)
                    ax_gt.set_title(f"GT: {mod_name}", fontsize=10)
                    ax_gt.axis("off")
                    col += 1

                    ax_syn = fig.add_subplot(gs[row_idx, col])
                    ax_syn.imshow(pred, cmap="gray", vmin=0, vmax=1)
                    p = psnr_numpy(pred, gt, data_range=1.0)
                    s = ssim_numpy(pred, gt, data_range=1.0)
                    ax_syn.set_title(f"Synth: {mod_name}\n{p:.1f}dB / {s:.3f}", fontsize=9)
                    ax_syn.axis("off")
                    col += 1

            while col <= 4:
                fig.add_subplot(gs[row_idx, col]).axis("off")
                col += 1

        status = "Trained" if trained else "UNTRAINED"
        fig.suptitle(f"All Scenarios - {name}  ({status})",
                     fontsize=16, fontweight="bold", y=0.98)
        save_path = output_dir / f"subject_grid_{name}.png"
        plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {save_path}")


# ============================================================
# Figure 3 & 4: PSNR and SSIM bar charts vs paper targets
# ============================================================

def fig_metric_bars(results, output_dir, trained):
    """PSNR and SSIM bar charts comparing to paper Table II."""
    scenarios_ordered = ["011", "101", "110", "001", "010", "100"]
    labels = [SCENARIO_LABELS.get(s, s) for s in scenarios_ordered]

    our_psnr = [results[s]["psnr_mean"] for s in scenarios_ordered]
    our_ssim = [results[s]["ssim_mean"] for s in scenarios_ordered]
    our_psnr_std = [results[s]["psnr_std"] for s in scenarios_ordered]
    our_ssim_std = [results[s]["ssim_std"] for s in scenarios_ordered]

    paper_psnr = [PAPER_TARGETS.get(s, {}).get("psnr_mean", None) for s in scenarios_ordered]
    paper_ssim = [PAPER_TARGETS.get(s, {}).get("ssim_mean", None) for s in scenarios_ordered]

    x = np.arange(len(scenarios_ordered))
    width = 0.35
    status = "Trained Model" if trained else "UNTRAINED Model"

    # ---- PSNR ----
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(x, our_psnr, width, yerr=our_psnr_std, capsize=4,
                  label=f"Our Results ({status})", color="#2196F3", alpha=0.85,
                  edgecolor="black", linewidth=0.5)

    first_paper = True
    for i, val in enumerate(paper_psnr):
        if val is not None:
            lbl = "Paper Table II" if first_paper else ""
            ax.hlines(val, i - width/2 - 0.05, i + width/2 + 0.05,
                      colors="red", linewidth=2.5, label=lbl)
            ax.scatter(i, val, color="red", s=80, zorder=5, marker="D")
            ax.annotate(f"{val:.1f}", (i, val), textcoords="offset points",
                        xytext=(25, 5), fontsize=9, color="red", fontweight="bold")
            first_paper = False

    for bar, val in zip(bars, our_psnr):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Missing Modality Scenario", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title(f"PSNR per Scenario  |  {status}", fontsize=14, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(axis="y", alpha=0.3); ax.set_ylim(bottom=0)
    plt.tight_layout()
    save_path = output_dir / "psnr_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")

    # ---- SSIM ----
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(x, our_ssim, width, yerr=our_ssim_std, capsize=4,
                  label=f"Our Results ({status})", color="#4CAF50", alpha=0.85,
                  edgecolor="black", linewidth=0.5)

    first_paper = True
    for i, val in enumerate(paper_ssim):
        if val is not None:
            lbl = "Paper Table II" if first_paper else ""
            ax.hlines(val, i - width/2 - 0.05, i + width/2 + 0.05,
                      colors="red", linewidth=2.5, label=lbl)
            ax.scatter(i, val, color="red", s=80, zorder=5, marker="D")
            ax.annotate(f"{val:.3f}", (i, val), textcoords="offset points",
                        xytext=(25, 5), fontsize=9, color="red", fontweight="bold")
            first_paper = False

    for bar, val in zip(bars, our_ssim):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Missing Modality Scenario", fontsize=12)
    ax.set_ylabel("SSIM", fontsize=12)
    ax.set_title(f"SSIM per Scenario  |  {status}", fontsize=14, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    save_path = output_dir / "ssim_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# Figure 5: Training loss curves
# ============================================================

def fig_training_curves(history, output_dir):
    """Plot G/D loss curves from checkpoint history."""
    if history is None:
        print("  [SKIP] No training history, generating placeholder")
        _fig_placeholder_training_curves(output_dir)
        return

    loss_G = history.get("train_loss_G", [])
    loss_D = history.get("train_loss_D", [])

    if not loss_G and not loss_D:
        print("  [SKIP] Training history empty, generating placeholder")
        _fig_placeholder_training_curves(output_dir)
        return

    epochs = range(1, len(loss_G) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, loss_G, "b-", linewidth=1.5, label="Generator Loss", alpha=0.8)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Generator Loss", fontsize=13, fontweight="bold")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, loss_D, "r-", linewidth=1.5, label="Discriminator Loss", alpha=0.8)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.set_title("Discriminator Loss", fontsize=13, fontweight="bold")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle(f"Training Loss Curves ({len(loss_G)} epochs)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    save_path = output_dir / "training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")


def _fig_placeholder_training_curves(output_dir):
    """Placeholder training curves when no history available."""
    np.random.seed(42)
    epochs = np.arange(1, 61)
    loss_G = 0.5 * np.exp(-0.03 * epochs) + 0.15 + np.random.normal(0, 0.02, len(epochs))
    loss_D = 0.3 * np.exp(-0.02 * epochs) + 0.2 + np.random.normal(0, 0.015, len(epochs))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, loss_G, "b-", linewidth=1.5, label="Generator Loss", alpha=0.8)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Generator Loss", fontsize=13, fontweight="bold")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax1.text(0.5, 0.95, "PLACEHOLDER", transform=ax1.transAxes, ha="center", va="top",
             fontsize=10, color="red", fontstyle="italic",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))

    ax2.plot(epochs, loss_D, "r-", linewidth=1.5, label="Discriminator Loss", alpha=0.8)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.set_title("Discriminator Loss", fontsize=13, fontweight="bold")
    ax2.legend(); ax2.grid(alpha=0.3)
    ax2.text(0.5, 0.95, "PLACEHOLDER", transform=ax2.transAxes, ha="center", va="top",
             fontsize=10, color="red", fontstyle="italic",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))

    fig.suptitle("Training Loss Curves (Placeholder)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    save_path = output_dir / "training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path} (placeholder)")


# ============================================================
# Figure 6: Validation metric curves with paper reference
# ============================================================

def fig_validation_curves(history, output_dir):
    """Validation PSNR/SSIM curves with paper reference lines."""
    if history is None:
        print("  [SKIP] No training history, generating placeholder")
        _fig_placeholder_validation_curves(output_dir)
        return

    val_psnr = history.get("val_psnr", [])
    val_ssim = history.get("val_ssim", [])

    if not val_psnr and not val_ssim:
        print("  [SKIP] Validation history empty, generating placeholder")
        _fig_placeholder_validation_curves(output_dir)
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    val_epochs = range(1, len(val_psnr) + 1)
    ax1.plot(val_epochs, val_psnr, "b-o", linewidth=1.5, markersize=5, label="Val PSNR", alpha=0.8)
    ax1.axhline(y=31.86, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
                label="Paper: T1,T2->PD (31.86)")
    ax1.axhline(y=31.32, color="orange", linestyle="--", linewidth=1.5, alpha=0.7,
                label="Paper: T2,PD->T1 (31.32)")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("PSNR (dB)")
    ax1.set_title("Validation PSNR", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9, loc="lower right"); ax1.grid(alpha=0.3)

    val_epochs = range(1, len(val_ssim) + 1)
    ax2.plot(val_epochs, val_ssim, "g-o", linewidth=1.5, markersize=5, label="Val SSIM", alpha=0.8)
    ax2.axhline(y=0.980, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
                label="Paper: T1,T2->PD (0.980)")
    ax2.axhline(y=0.948, color="orange", linestyle="--", linewidth=1.5, alpha=0.7,
                label="Paper: T2,PD->T1 (0.948)")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("SSIM")
    ax2.set_title("Validation SSIM", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9, loc="lower right"); ax2.grid(alpha=0.3)

    fig.suptitle(f"Validation Metrics vs Paper Targets ({len(val_psnr)} epochs)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    save_path = output_dir / "validation_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")


def _fig_placeholder_validation_curves(output_dir):
    """Placeholder validation curves."""
    np.random.seed(42)
    steps = np.arange(1, 31)
    psnr = 18 + 10 * (1 - np.exp(-0.08 * steps)) + np.random.normal(0, 0.3, len(steps))
    ssim = 0.6 + 0.3 * (1 - np.exp(-0.07 * steps)) + np.random.normal(0, 0.01, len(steps))
    ssim = np.clip(ssim, 0, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(steps, psnr, "b-o", linewidth=1.5, markersize=4, label="Val PSNR", alpha=0.8)
    ax1.axhline(y=31.86, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
                label="Paper: T1,T2->PD (31.86)")
    ax1.axhline(y=31.32, color="orange", linestyle="--", linewidth=1.5, alpha=0.7,
                label="Paper: T2,PD->T1 (31.32)")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("PSNR (dB)")
    ax1.set_title("Validation PSNR", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)
    ax1.text(0.5, 0.05, "PLACEHOLDER", transform=ax1.transAxes, ha="center",
             fontsize=12, color="red", fontstyle="italic",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))

    ax2.plot(steps, ssim, "g-o", linewidth=1.5, markersize=4, label="Val SSIM", alpha=0.8)
    ax2.axhline(y=0.980, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
                label="Paper: T1,T2->PD (0.980)")
    ax2.axhline(y=0.948, color="orange", linestyle="--", linewidth=1.5, alpha=0.7,
                label="Paper: T2,PD->T1 (0.948)")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("SSIM")
    ax2.set_title("Validation SSIM", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
    ax2.text(0.5, 0.05, "PLACEHOLDER", transform=ax2.transAxes, ha="center",
             fontsize=12, color="red", fontstyle="italic",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))

    fig.suptitle("Validation Metrics vs Paper Targets (Placeholder)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    save_path = output_dir / "validation_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path} (placeholder)")


# ============================================================
# Console + file: Results table
# ============================================================

def print_results_table(results, trained, output_dir=None):
    """Print formatted results table and optionally save to file."""
    status = "TRAINED" if trained else "UNTRAINED"
    order = ["011", "101", "110", "001", "010", "100"]

    lines = []
    lines.append(f"{'='*90}")
    lines.append(f"  EVALUATION RESULTS  ({status} model)")
    lines.append(f"{'='*90}")
    lines.append(
        f"{'Scenario':<12} {'Available':<12} {'Missing':<12} "
        f"{'PSNR':>8} {'(+/-)':>6} {'SSIM':>8} {'(+/-)':>7} "
        f"{'Paper PSNR':>11} {'Paper SSIM':>11}")
    lines.append(f"{'-'*90}")

    for s in order:
        r = results[s]
        avail = ",".join(r["available"])
        miss = ",".join(r["missing"])
        pp = PAPER_TARGETS.get(s, {}).get("psnr_mean")
        ps = PAPER_TARGETS.get(s, {}).get("ssim_mean")
        lines.append(
            f"{s:<12} {avail:<12} {miss:<12} "
            f"{r['psnr_mean']:>8.2f} {r['psnr_std']:>5.2f} "
            f"{r['ssim_mean']:>8.4f} {r['ssim_std']:>6.4f} "
            f"{pp if pp else '--':>11} {ps if ps else '--':>11}")

    all_psnr = [results[s]["psnr_mean"] for s in order]
    all_ssim = [results[s]["ssim_mean"] for s in order]
    lines.append(f"{'-'*90}")
    lines.append(
        f"{'OVERALL':<12} {'':12} {'':12} "
        f"{np.mean(all_psnr):>8.2f} {'':>6} "
        f"{np.mean(all_ssim):>8.4f}")
    lines.append(f"{'='*90}")

    text = "\n".join(lines)
    print(f"\n{text}")

    if output_dir:
        save_path = Path(output_dir) / "results_table.txt"
        with open(save_path, "w") as f:
            f.write(text + "\n")
        print(f"  Saved: {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate all presentation figures for MM-GAN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # RECOMMENDED: real test data + trained model
  python generate_presentation.py \\
      --data_dir ../data/processed/baseline \\
      --checkpoint ./comparer/baseline/checkpoints/baseline

  # Quick run (fewer slices for faster execution):
  python generate_presentation.py \\
      --data_dir ../data/processed/baseline \\
      --checkpoint ./comparer/baseline/checkpoints/baseline \\
      --n_eval 100

  # Use pre-computed metrics (skip expensive inference):
  python generate_presentation.py \\
      --data_dir ../data/processed/baseline \\
      --checkpoint ./comparer/baseline/checkpoints/baseline \\
      --metrics_json ./comparer/baseline/results/baseline/metrics.json

  # Demo mode (synthetic data, no model needed):
  python generate_presentation.py --demo
        """,
    )

    parser.add_argument("--demo", action="store_true",
                        help="Demo mode: synthetic brain phantoms (no download)")
    parser.add_argument("--ixi_dir", type=str, default=None,
                        help="Directory with raw IXI .nii.gz files")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Root directory with train/val/test .npy slices")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file or directory")
    parser.add_argument("--metrics_json", type=str, default=None,
                        help="Pre-computed metrics.json (skip inference for metrics)")
    parser.add_argument("--n_visual", type=int, default=5,
                        help="Number of subjects for visual figures")
    parser.add_argument("--n_eval", type=int, default=200,
                        help="Number of slices for metrics (0 = all)")
    parser.add_argument("--output_dir", type=str, default="./presentation",
                        help="Directory to save all figures")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto', 'cuda', or 'cpu'")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    if not args.demo and args.data_dir is None and args.ixi_dir is None:
        parser.error("One of --demo, --ixi_dir, or --data_dir must be provided")

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove previously generated files so the folder stays fresh on each run
    removed = 0
    for pattern in ("*.png", "*.txt", "*.json"):
        for old_file in output_dir.glob(pattern):
            old_file.unlink()
            removed += 1
    if removed:
        print(f"  Cleared {removed} old file(s) from {output_dir}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    mode_str = ("DEMO (synthetic)" if args.demo
                else ("IXI raw NIfTI" if args.ixi_dir else "Processed .npy"))

    print("=" * 60)
    print("  MM-GAN Presentation Figure Generator")
    print("=" * 60)
    print(f"  Mode:       {mode_str}")
    print(f"  Device:     {device}")
    print(f"  Checkpoint: {args.checkpoint or 'None (untrained)'}")
    print(f"  Output:     {output_dir}")
    if args.metrics_json:
        print(f"  Metrics:    {args.metrics_json} (pre-computed)")
    print()

    # ---- Step 1: Load model ----
    print("[1/7] Loading model...")
    generator, trained, epoch_info, history = load_model(args.checkpoint, device)

    # ---- Step 2: Load data for VISUALS (few subjects, best slices) ----
    print(f"\n[2/7] Loading data for visual figures ({args.n_visual} subjects)...")
    if args.demo:
        visual_slices = generate_demo_subjects(n_subjects=args.n_visual)
    elif args.ixi_dir:
        visual_slices = load_raw_nifti_slices(args.ixi_dir, n_subjects=args.n_visual)
    else:
        visual_slices = load_npy_slices_for_visual(args.data_dir, n_subjects=args.n_visual)

    print(f"  Loaded {len(visual_slices)} subjects for visuals")

    # ---- Step 3: Run inference on visual subjects ----
    print(f"\n[3/7] Running inference on visual subjects...")
    _, _, visual_synth = evaluate_all_scenarios(
        generator, visual_slices, device, verbose=False)

    # ---- Step 4: Generate visual figures ----
    print(f"\n[4/7] Generating side-by-side visual comparisons...")
    fig_visual_comparison(visual_slices, visual_synth, output_dir, epoch_info if trained else False)

    print(f"\n[5/7] Generating per-subject scenario grids...")
    fig_subject_grid(visual_slices, visual_synth, output_dir, trained)

    # ---- Step 5: Quantitative evaluation ----
    print(f"\n[6/7] Quantitative evaluation...")

    loaded = None
    if args.metrics_json and Path(args.metrics_json).exists():
        # Use pre-computed metrics
        print(f"  Loading pre-computed metrics from {args.metrics_json}")
        try:
            with open(args.metrics_json) as f:
                content = f.read().strip()
                if not content:
                    raise ValueError("metrics JSON file is empty")
                loaded = json.loads(content)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  WARNING: Could not load metrics from {args.metrics_json}: {e}")
            print(f"  Falling back to on-the-fly evaluation...")
            args.metrics_json = None
            loaded = None

    if args.metrics_json and loaded is not None:

        # Normalize format: the pre-computed file may have different structure
        if "per_scenario" in loaded:
            results = loaded["per_scenario"]
        else:
            results = loaded

        # Ensure 'available' and 'missing' keys exist
        for s_str, r in results.items():
            scenario = [int(c) for c in s_str]
            if "available" not in r:
                r["available"] = [MODALITY_NAMES[i] for i, s in enumerate(scenario) if s == 1]
            if "missing" not in r:
                r["missing"] = [MODALITY_NAMES[i] for i, s in enumerate(scenario) if s == 0]
            if "psnr_std" not in r:
                r["psnr_std"] = 0.0
            if "ssim_std" not in r:
                r["ssim_std"] = 0.0

        print(f"  Loaded metrics for {len(results)} scenarios")
    if args.metrics_json is None or loaded is None:
        # Run evaluation on test slices
        if args.demo:
            eval_slices = visual_slices
        elif args.ixi_dir:
            eval_slices = visual_slices
        else:
            print(f"  Loading evaluation slices (n_eval={args.n_eval})...")
            eval_slices = load_npy_slices_for_eval(args.data_dir, n_slices=args.n_eval)

        print(f"  Evaluating on {len(eval_slices)} slices...")
        results, per_sample, _ = evaluate_all_scenarios(generator, eval_slices, device)

    # Print & save results table
    print_results_table(results, trained, output_dir)

    # Save metrics JSON
    metrics_path = output_dir / "metrics.json"
    serializable = {}
    for k, v in results.items():
        sv = dict(v)
        if "scenario" in sv and isinstance(sv["scenario"], list):
            sv["scenario"] = "".join(str(x) for x in sv["scenario"])
        serializable[k] = sv
    with open(metrics_path, "w") as f:
        json.dump({
            "per_scenario": serializable,
            "overall": {
                "psnr_mean": float(np.mean([r["psnr_mean"] for r in results.values()])),
                "ssim_mean": float(np.mean([r["ssim_mean"] for r in results.values()])),
            },
            "trained": trained,
            "epoch": str(epoch_info),
        }, f, indent=2)
    print(f"  Saved: {metrics_path}")

    # ---- Step 6: Metric bar charts ----
    print(f"\n[7/7] Generating charts...")
    fig_metric_bars(results, output_dir, trained)

    # Training/validation curves
    print("\n  Training and validation curves...")
    fig_training_curves(history, output_dir)
    fig_validation_curves(history, output_dir)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  ALL FIGURES GENERATED SUCCESSFULLY")
    print("=" * 60)
    print(f"\n  Output directory: {output_dir}")
    print(f"\n  Files:")
    for f in sorted(output_dir.glob("*")):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:45s} ({size_kb:7.1f} KB)")

    if not trained:
        print(f"\n  NOTE: Results are from an UNTRAINED model (random weights).")
        print(f"  Re-run with --checkpoint after training to get real results.")

    print()


if __name__ == "__main__":
    main()
