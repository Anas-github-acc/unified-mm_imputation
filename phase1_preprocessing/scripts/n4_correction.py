#!/usr/bin/env python3
"""
Step 3: N4 Bias Field Correction (Optimization Step)
Applies N4 bias field correction to all registered volumes.
This is the key optimization step that differentiates baseline vs optimized.

Run: python n4_correction.py --input_dir ../data/processed/registered
                              --output_dir ../data/processed/n4_corrected
"""

import os
import sys
import argparse
import time
from pathlib import Path

import ants
import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm


def apply_n4_correction(
    image_path: str,
    output_path: str,
    shrink_factor: int = 4,
    convergence_iters: list = None,
    convergence_tol: float = 0.0,
):
    """
    Apply N4 bias field correction to a single NIfTI volume.
    Returns the corrected image path.
    """
    if convergence_iters is None:
        convergence_iters = [50, 50, 50, 50]

    if os.path.exists(output_path):
        return output_path, True  # skipped

    img = ants.image_read(image_path)

    # Create a brain mask (simple threshold-based)
    arr = img.numpy()
    threshold = np.percentile(arr[arr > 0], 5) if np.any(arr > 0) else 0
    mask_arr = (arr > threshold).astype(np.uint8)
    mask = ants.from_numpy(mask_arr, origin=img.origin, spacing=img.spacing, direction=img.direction)

    # Apply N4 correction
    corrected = ants.n4_bias_field_correction(
        img,
        mask=mask,
        shrink_factor=shrink_factor,
        convergence={"iters": convergence_iters, "tol": convergence_tol},
        verbose=False,
    )

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ants.image_write(corrected, output_path)

    return output_path, False


def main():
    parser = argparse.ArgumentParser(description="N4 Bias Field Correction")
    parser.add_argument("--input_dir", type=str, default="data/processed/registered")
    parser.add_argument("--output_dir", type=str, default="data/processed/n4_corrected")
    parser.add_argument("--config", type=str, default="phase1_preprocessing/configs/config.yaml")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--max_subjects", type=int, default=-1)
    args = parser.parse_args()

    # Load config
    config = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

    n4_cfg = config.get("n4", {})
    shrink_factor = n4_cfg.get("shrink_factor", 4)
    convergence = n4_cfg.get("convergence", {})
    conv_iters = convergence.get("iters", [50, 50, 50, 50])
    conv_tol = convergence.get("tol", 0.0)

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("N4 Bias Field Correction Pipeline")
    print("=" * 60)
    print(f"Shrink factor:     {shrink_factor}")
    print(f"Convergence iters: {conv_iters}")
    print(f"Input:             {input_dir}")
    print(f"Output:            {output_dir}")

    # Find all subject directories
    subject_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("IXI")])

    if args.max_subjects > 0:
        subject_dirs = subject_dirs[args.start_idx:args.start_idx + args.max_subjects]
    else:
        subject_dirs = subject_dirs[args.start_idx:]

    print(f"Subjects to process: {len(subject_dirs)}")
    print()

    modality_files = {
        "T1": "T1_registered.nii.gz",
        "T2": "T2_registered.nii.gz",
        "PD": "PD.nii.gz",
    }

    results = []
    failed = []
    start_time = time.time()

    for subj_dir in tqdm(subject_dirs, desc="N4 Correction"):
        sid = subj_dir.name
        out_subj = output_dir / sid
        out_subj.mkdir(parents=True, exist_ok=True)

        subj_result = {"subject_id": sid}
        subj_ok = True

        for mod_name, filename in modality_files.items():
            input_path = subj_dir / filename
            # Keep same filenames in output
            output_path = out_subj / filename

            if not input_path.exists():
                tqdm.write(f"  [WARN] {sid}: Missing {filename}")
                subj_ok = False
                continue

            try:
                _, skipped = apply_n4_correction(
                    str(input_path),
                    str(output_path),
                    shrink_factor=shrink_factor,
                    convergence_iters=conv_iters,
                    convergence_tol=conv_tol,
                )
                subj_result[f"{mod_name.lower()}_n4"] = str(output_path)
                status = "SKIP" if skipped else "OK"

            except Exception as e:
                tqdm.write(f"  [FAIL] {sid}/{mod_name}: {e}")
                subj_ok = False
                failed.append({"subject_id": sid, "modality": mod_name, "error": str(e)})

        if subj_ok:
            results.append(subj_result)

    elapsed = time.time() - start_time

    # Save N4 manifest
    n4_df = pd.DataFrame(results)
    n4_manifest_path = output_dir / "n4_correction_manifest.csv"
    n4_df.to_csv(n4_manifest_path, index=False)

    print("\n" + "=" * 60)
    print("N4 Correction Summary")
    print(f"  Successful: {len(results)}")
    print(f"  Failed:     {len(failed)}")
    print(f"  Time:       {elapsed:.1f}s ({elapsed/max(len(subject_dirs),1):.1f}s/subject)")
    print(f"  Output:     {output_dir}")
    print("=" * 60)
    print("\nNext step: Run 'python extract_slices.py' for slice extraction.")


if __name__ == "__main__":
    main()
