#!/usr/bin/env python3
"""
Step 5: Quality Control Visualization
Generates comparison images for QC:
  - Raw vs Registered vs N4-corrected
  - Side-by-side modality comparisons
  - Intensity histograms before/after N4

Run: python qc_visualize.py --registered_dir ../data/processed/registered
                              --n4_dir ../data/processed/n4_corrected
                              --output_dir ../artifacts/qc
"""

import os
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm


def load_middle_slice(nifti_path: str, axis: int = 2) -> np.ndarray:
    """Load a NIfTI file and return the middle axial slice."""
    img = nib.load(nifti_path)
    data = img.get_fdata().astype(np.float32)
    mid = data.shape[axis] // 2
    if axis == 0:
        return data[mid, :, :]
    elif axis == 1:
        return data[:, mid, :]
    else:
        return data[:, :, mid]


def plot_registration_comparison(
    raw_paths: dict,
    reg_paths: dict,
    subject_id: str,
    output_path: str,
):
    """
    Plot raw vs registered modalities for one subject.
    Shows T1, T2, PD before and after registration.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Registration QC: {subject_id}", fontsize=16, fontweight="bold")

    modalities = ["T1", "T2", "PD"]
    row_labels = ["Raw", "Registered"]

    for col, mod in enumerate(modalities):
        # Raw
        if mod in raw_paths and os.path.exists(raw_paths[mod]):
            raw_sl = load_middle_slice(raw_paths[mod])
            axes[0, col].imshow(raw_sl.T, cmap="gray", origin="lower")
        axes[0, col].set_title(f"{mod} (Raw)")
        axes[0, col].axis("off")

        # Registered
        if mod in reg_paths and os.path.exists(reg_paths[mod]):
            reg_sl = load_middle_slice(reg_paths[mod])
            axes[1, col].imshow(reg_sl.T, cmap="gray", origin="lower")
        axes[1, col].set_title(f"{mod} (Registered)")
        axes[1, col].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_n4_comparison(
    reg_paths: dict,
    n4_paths: dict,
    subject_id: str,
    output_path: str,
):
    """
    Plot registered vs N4-corrected for one subject.
    Shows the optimization improvement.
    """
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 6, figure=fig, hspace=0.3, wspace=0.1)
    fig.suptitle(f"N4 Bias Correction QC: {subject_id}", fontsize=16, fontweight="bold")

    modalities = ["T1", "T2", "PD"]

    for col, mod in enumerate(modalities):
        reg_file = reg_paths.get(mod, "")
        n4_file = n4_paths.get(mod, "")

        if os.path.exists(reg_file):
            reg_sl = load_middle_slice(reg_file)
        else:
            reg_sl = np.zeros((100, 100))

        if os.path.exists(n4_file):
            n4_sl = load_middle_slice(n4_file)
        else:
            n4_sl = np.zeros((100, 100))

        # Before N4
        ax1 = fig.add_subplot(gs[0, col * 2])
        ax1.imshow(reg_sl.T, cmap="gray", origin="lower")
        ax1.set_title(f"{mod}\nBefore N4")
        ax1.axis("off")

        # After N4
        ax2 = fig.add_subplot(gs[0, col * 2 + 1])
        ax2.imshow(n4_sl.T, cmap="gray", origin="lower")
        ax2.set_title(f"{mod}\nAfter N4")
        ax2.axis("off")

        # Difference map
        ax3 = fig.add_subplot(gs[1, col * 2:col * 2 + 2])
        diff = np.abs(n4_sl - reg_sl)
        ax3.imshow(diff.T, cmap="hot", origin="lower")
        ax3.set_title(f"{mod} |Difference|")
        ax3.axis("off")

        # Intensity histogram
        ax4 = fig.add_subplot(gs[2, col * 2:col * 2 + 2])
        reg_flat = reg_sl[reg_sl > 0].flatten()
        n4_flat = n4_sl[n4_sl > 0].flatten()
        if len(reg_flat) > 0:
            ax4.hist(reg_flat, bins=100, alpha=0.5, label="Before N4", density=True, color="blue")
        if len(n4_flat) > 0:
            ax4.hist(n4_flat, bins=100, alpha=0.5, label="After N4", density=True, color="orange")
        ax4.set_title(f"{mod} Intensity Distribution")
        ax4.legend(fontsize=8)
        ax4.set_xlabel("Intensity")
        ax4.set_ylabel("Density")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="QC Visualization")
    parser.add_argument("--raw_dir", type=str, default="data/raw/ixi")
    parser.add_argument("--registered_dir", type=str, default="data/processed/registered")
    parser.add_argument("--n4_dir", type=str, default="data/processed/n4_corrected")
    parser.add_argument("--manifest", type=str, default="data/manifests/subject_manifest.csv")
    parser.add_argument("--output_dir", type=str, default="artifacts/qc")
    parser.add_argument("--max_subjects", type=int, default=5, help="Number of subjects to visualize")
    args = parser.parse_args()

    import pandas as pd
    df = pd.read_csv(args.manifest)

    output_dir = Path(args.output_dir)
    (output_dir / "registration").mkdir(parents=True, exist_ok=True)
    (output_dir / "n4_correction").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("QC Visualization")
    print("=" * 60)

    raw_dir = Path(args.raw_dir)
    reg_dir = Path(args.registered_dir)
    n4_dir = Path(args.n4_dir)

    # Sample subjects from each split
    subjects = df.head(args.max_subjects)

    for _, row in tqdm(subjects.iterrows(), total=len(subjects), desc="Generating QC"):
        sid = row["subject_id"]

        # Build paths
        raw_paths = {
            "T1": row.get("t1_path", ""),
            "T2": row.get("t2_path", ""),
            "PD": row.get("pd_path", ""),
        }

        reg_paths = {
            "T1": str(reg_dir / sid / "T1_registered.nii.gz"),
            "T2": str(reg_dir / sid / "T2_registered.nii.gz"),
            "PD": str(reg_dir / sid / "PD.nii.gz"),
        }

        n4_paths = {
            "T1": str(n4_dir / sid / "T1_registered.nii.gz"),
            "T2": str(n4_dir / sid / "T2_registered.nii.gz"),
            "PD": str(n4_dir / sid / "PD.nii.gz"),
        }

        # Registration QC
        try:
            plot_registration_comparison(
                raw_paths, reg_paths, sid,
                str(output_dir / "registration" / f"{sid}_registration_qc.png"),
            )
        except Exception as e:
            tqdm.write(f"  [WARN] Registration QC failed for {sid}: {e}")

        # N4 QC
        try:
            if (n4_dir / sid).exists():
                plot_n4_comparison(
                    reg_paths, n4_paths, sid,
                    str(output_dir / "n4_correction" / f"{sid}_n4_qc.png"),
                )
        except Exception as e:
            tqdm.write(f"  [WARN] N4 QC failed for {sid}: {e}")

    print(f"\nQC images saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
