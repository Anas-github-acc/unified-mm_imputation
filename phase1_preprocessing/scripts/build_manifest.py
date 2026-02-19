#!/usr/bin/env python3
"""
Step 1: Build Subject Manifest
Scans the IXI raw directory, matches T1/T2/PD by subject ID, and creates
a manifest CSV with only subjects that have ALL 3 modalities.
Also creates the train/val/test split.

Run: python build_manifest.py --ixi_dir ../data/raw/ixi --output ../data/manifests
"""

import os
import re
import argparse
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd
import yaml


def parse_ixi_filename(filename: str):
    """
    Parse IXI filename to extract subject ID, site, and modality.
    Format: IXI{ID}-{Site}-{ScanID}-{Modality}.nii.gz
    Example: IXI002-Guys-0828-T1.nii.gz
    """
    # Handle both .nii.gz and .nii
    basename = filename.replace(".nii.gz", "").replace(".nii", "")
    parts = basename.split("-")
    if len(parts) < 4:
        return None

    subject_id = parts[0]  # e.g., IXI002
    site = parts[1]         # e.g., Guys
    scan_id = parts[2]      # e.g., 0828
    modality = parts[3]     # e.g., T1

    return {
        "subject_id": subject_id,
        "site": site,
        "scan_id": scan_id,
        "modality": modality,
    }


def build_manifest(ixi_dir: str, output_dir: str, config: dict):
    """Build subject manifest and create train/val/test split."""
    ixi_path = Path(ixi_dir).resolve()
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Building Subject Manifest")
    print("=" * 60)
    print(f"IXI directory: {ixi_path}")

    # Collect all NIfTI files
    nifti_files = list(ixi_path.glob("IXI*.nii.gz")) + list(ixi_path.glob("IXI*.nii"))
    print(f"Total NIfTI files found: {len(nifti_files)}")

    # Group by subject
    subjects = defaultdict(dict)
    for filepath in nifti_files:
        parsed = parse_ixi_filename(filepath.name)
        if parsed and parsed["modality"] in ("T1", "T2", "PD"):
            sid = parsed["subject_id"]
            mod = parsed["modality"]
            subjects[sid][mod] = str(filepath)
            subjects[sid]["site"] = parsed["site"]

    # Filter to subjects with all 3 modalities
    complete_subjects = {}
    for sid, data in subjects.items():
        if all(m in data for m in ("T1", "T2", "PD")):
            complete_subjects[sid] = data

    print(f"Subjects with all 3 modalities (T1, T2, PD): {len(complete_subjects)}")

    # Build dataframe
    rows = []
    for sid in sorted(complete_subjects.keys()):
        data = complete_subjects[sid]
        rows.append({
            "subject_id": sid,
            "site": data["site"],
            "t1_path": data["T1"],
            "t2_path": data["T2"],
            "pd_path": data["PD"],
        })

    df = pd.DataFrame(rows)

    # Create train/val/test split
    split_cfg = config.get("split", {})
    seed = split_cfg.get("random_seed", 42)
    train_r = split_cfg.get("train_ratio", 0.85)
    val_r = split_cfg.get("val_ratio", 0.075)
    test_r = split_cfg.get("test_ratio", 0.075)

    random.seed(seed)
    indices = list(range(len(df)))
    random.shuffle(indices)

    n = len(df)
    n_train = int(n * train_r)
    n_val = int(n * val_r)

    splits = [""] * n
    for i, idx in enumerate(indices):
        if i < n_train:
            splits[idx] = "train"
        elif i < n_train + n_val:
            splits[idx] = "val"
        else:
            splits[idx] = "test"

    df["split"] = splits

    # Save full manifest
    manifest_path = output_path / "subject_manifest.csv"
    df.to_csv(manifest_path, index=False)
    print(f"\nManifest saved: {manifest_path}")

    # Summary
    print("\nSplit Summary:")
    print(f"  Train: {(df['split'] == 'train').sum()}")
    print(f"  Val:   {(df['split'] == 'val').sum()}")
    print(f"  Test:  {(df['split'] == 'test').sum()}")
    print(f"  Total: {len(df)}")

    # Site distribution
    print("\nSite Distribution:")
    for site, count in df["site"].value_counts().items():
        print(f"  {site}: {count}")

    print("=" * 60)
    print("\nNext step: Run 'python register.py' to register volumes.")

    return df


def main():
    parser = argparse.ArgumentParser(description="Build IXI subject manifest")
    parser.add_argument("--ixi_dir", type=str, default="../data/raw/ixi")
    parser.add_argument("--output", type=str, default="../data/manifests")
    parser.add_argument("--config", type=str, default="../phase1_preprocessing/configs/config.yaml")
    args = parser.parse_args()

    # Load config
    config = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

    build_manifest(args.ixi_dir, args.output, config)


if __name__ == "__main__":
    main()
