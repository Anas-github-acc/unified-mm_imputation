#!/usr/bin/env python3
"""
Step 2: Registration Pipeline (ANTsPy Affine)
Registers T1 and T2 to PD space for each subject using ANTsPy.
Saves registered volumes as NIfTI.

Run: python register.py --manifest ../data/manifests/subject_manifest.csv
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Set ITK threads before importing ants
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"

import ants
import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm


def register_subject(
    subject_id: str,
    t1_path: str,
    t2_path: str,
    pd_path: str,
    output_dir: str,
    transform_type: str = "Affine",
    verbose: bool = False,
):
    """
    Register T1 and T2 to PD space for a single subject.
    PD is the fixed image (reference space).
    Returns dict of output paths.
    """
    out = Path(output_dir) / subject_id
    out.mkdir(parents=True, exist_ok=True)

    # Check if already done
    t1_out = out / "T1_registered.nii.gz"
    t2_out = out / "T2_registered.nii.gz"
    pd_out = out / "PD.nii.gz"

    if t1_out.exists() and t2_out.exists() and pd_out.exists():
        return {
            "t1_registered": str(t1_out),
            "t2_registered": str(t2_out),
            "pd": str(pd_out),
            "skipped": True,
        }

    # Load images
    fixed = ants.image_read(pd_path)   # PD = reference space
    t1 = ants.image_read(t1_path)
    t2 = ants.image_read(t2_path)

    # Register T1 -> PD
    tx_t1 = ants.registration(
        fixed=fixed,
        moving=t1,
        type_of_transform=transform_type,
        verbose=verbose,
    )
    t1_warped = tx_t1["warpedmovout"]

    # Register T2 -> PD
    tx_t2 = ants.registration(
        fixed=fixed,
        moving=t2,
        type_of_transform=transform_type,
        verbose=verbose,
    )
    t2_warped = tx_t2["warpedmovout"]

    # Save registered volumes
    ants.image_write(t1_warped, str(t1_out))
    ants.image_write(t2_warped, str(t2_out))
    ants.image_write(fixed, str(pd_out))  # Save PD as-is (it's the reference)

    return {
        "t1_registered": str(t1_out),
        "t2_registered": str(t2_out),
        "pd": str(pd_out),
        "skipped": False,
    }


def main():
    parser = argparse.ArgumentParser(description="Register IXI volumes (T1/T2 -> PD)")
    parser.add_argument("--manifest", type=str, default="../data/manifests/subject_manifest.csv")
    parser.add_argument("--output_dir", type=str, default="../data/processed/registered")
    parser.add_argument("--config", type=str, default="../phase1_preprocessing/configs/config.yaml")
    parser.add_argument("--start_idx", type=int, default=0, help="Start from this subject index")
    parser.add_argument("--max_subjects", type=int, default=-1, help="Process at most N subjects (-1=all)")
    parser.add_argument("--n_threads", type=int, default=None, help="Number of threads for ITK/ANTs (default: from config or 8)")
    args = parser.parse_args()

    # Load config
    config = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

    reg_cfg = config.get("registration", {})
    transform_type = reg_cfg.get("type_of_transform", "Affine")
    verbose = reg_cfg.get("verbose", False)
    
    # Set number of threads (priority: CLI arg > config > default 8)
    n_threads = args.n_threads if args.n_threads is not None else reg_cfg.get("n_threads", 8)
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(n_threads)

    # Load manifest
    df = pd.read_csv(args.manifest)
    print("=" * 60)
    print("ANTsPy Registration Pipeline")
    print("=" * 60)
    print(f"Transform type: {transform_type}")
    print(f"Number of threads: {n_threads}")
    print(f"Total subjects in manifest: {len(df)}")

    # Subset if requested
    if args.max_subjects > 0:
        df = df.iloc[args.start_idx:args.start_idx + args.max_subjects]
    else:
        df = df.iloc[args.start_idx:]

    print(f"Processing subjects {args.start_idx} to {args.start_idx + len(df) - 1}")
    print()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process subjects
    results = []
    failed = []
    start_time = time.time()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Registering"):
        sid = row["subject_id"]
        try:
            result = register_subject(
                subject_id=sid,
                t1_path=row["t1_path"],
                t2_path=row["t2_path"],
                pd_path=row["pd_path"],
                output_dir=str(output_dir),
                transform_type=transform_type,
                verbose=verbose,
            )
            result["subject_id"] = sid
            results.append(result)

            status = "SKIP" if result.get("skipped") else "OK"
            tqdm.write(f"  [{status}] {sid}")

        except Exception as e:
            tqdm.write(f"  [FAIL] {sid}: {e}")
            failed.append({"subject_id": sid, "error": str(e)})

    elapsed = time.time() - start_time

    # Save registration manifest
    reg_df = pd.DataFrame(results)
    reg_manifest_path = output_dir / "registration_manifest.csv"
    reg_df.to_csv(reg_manifest_path, index=False)

    print("\n" + "=" * 60)
    print("Registration Summary")
    print(f"  Successful: {len(results)}")
    print(f"  Failed:     {len(failed)}")
    print(f"  Time:       {elapsed:.1f}s ({elapsed/max(len(df),1):.1f}s/subject)")
    print(f"  Output:     {output_dir}")
    print(f"  Manifest:   {reg_manifest_path}")

    if failed:
        print("\nFailed subjects:")
        for f in failed:
            print(f"  {f['subject_id']}: {f['error']}")
        failed_df = pd.DataFrame(failed)
        failed_path = output_dir / "registration_failed.csv"
        failed_df.to_csv(failed_path, index=False)
        print(f"  Saved to: {failed_path}")

    print("=" * 60)
    print("\nNext step: Run 'python n4_correction.py' for bias field correction.")


if __name__ == "__main__":
    main()
