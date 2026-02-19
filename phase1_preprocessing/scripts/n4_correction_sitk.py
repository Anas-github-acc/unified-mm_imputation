#!/usr/bin/env python3
"""
Step 3: N4 Bias Field Correction (Optimization Step) - SimpleITK Version
Applies N4 bias field correction using SimpleITK instead of ANTs.
This version is more stable and avoids segmentation faults.

Run: python n4_correction_sitk.py --input_dir ../data/processed/registered
                                   --output_dir ../data/processed/n4_corrected
"""

import os
import sys
import argparse
import time
from pathlib import Path

import SimpleITK as sitk
import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm


def apply_n4_correction_sitk(
    image_path: str,
    output_path: str,
    shrink_factor: int = 4,
    num_iterations: list = None,
    num_fitting_levels: int = 4,
):
    """
    Apply N4 bias field correction using SimpleITK.
    Returns the corrected image path and whether it was skipped.
    """
    if num_iterations is None:
        num_iterations = [50, 50, 50, 50]

    if os.path.exists(output_path):
        return output_path, True  # skipped

    ##print(f"[DEBUG] Reading image: {image_path}", flush=True)
    # Read the image
    image = sitk.ReadImage(image_path, sitk.sitkFloat32)
    #print(f"[DEBUG] Image read successfully, size: {image.GetSize()}", flush=True)

    # Create a simple mask based on Otsu thresholding
    #print(f"[DEBUG] Creating mask using Otsu thresholding...", flush=True)
    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    #print(f"[DEBUG] Mask created", flush=True)

    # Shrink the image for faster processing
    if shrink_factor > 1:
        #print(f"[DEBUG] Shrinking image by factor {shrink_factor}...", flush=True)
        image_shrunk = sitk.Shrink(image, [shrink_factor] * image.GetDimension())
        mask_shrunk = sitk.Shrink(mask, [shrink_factor] * mask.GetDimension())
    else:
        image_shrunk = image
        mask_shrunk = mask

    # Set up N4 bias field corrector
    #print(f"[DEBUG] Setting up N4 bias field corrector...", flush=True)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(num_iterations)

    # Execute correction on shrunk image
    #print(f"[DEBUG] Executing N4 bias field correction...", flush=True)
    try:
        corrected_shrunk = corrector.Execute(image_shrunk, mask_shrunk)
        #print(f"[DEBUG] N4  correction completed on shrunk image", flush=True)
    except Exception as e:
        print(f"[ERROR] N4 correction failed: {e}", flush=True)
        raise

    # Get the bias field and apply it to the full resolution image
    if shrink_factor > 1:
        #print(f"[DEBUG] Extracting and upsampling bias field...", flush=True)
        log_bias_field = corrector.GetLogBiasFieldAsImage(image_shrunk)
        
        # Resample the bias field back to original size
        log_bias_field_full = sitk.Resample(
            log_bias_field,
            image,
            sitk.Transform(),
            sitk.sitkLinear,
            0.0,
            log_bias_field.GetPixelID()
        )
        
        # Apply bias field correction to full resolution image
        corrected = image / sitk.Exp(log_bias_field_full)
        #print(f"[DEBUG] Bias field applied to full resolution image", flush=True)
    else:
        corrected = corrected_shrunk

    # Save the corrected image
    #print(f"[DEBUG] Saving corrected image to: {output_path}", flush=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sitk.WriteImage(corrected, output_path)
    #print(f"[DEBUG] Image saved successfully", flush=True)

    return output_path, False


def main():
    parser = argparse.ArgumentParser(description="N4 Bias Field Correction (SimpleITK)")
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

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("N4 Bias Field Correction Pipeline (SimpleITK)")
    print("=" * 60)
    print(f"Shrink factor:     {shrink_factor}")
    print(f"Iterations:        {conv_iters}")
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

            # print(f"\n[DEBUG] Processing {sid}/{mod_name}: {input_path}", flush=True)
            try:
                _, skipped = apply_n4_correction_sitk(
                    str(input_path),
                    str(output_path),
                    shrink_factor=shrink_factor,
                    num_iterations=conv_iters,
                )
                #print(f"[DEBUG] Completed {sid}/{mod_name}", flush=True)
                subj_result[f"{mod_name.lower()}_n4"] = str(output_path)
                status = "SKIP" if skipped else "OK"
            except Exception as e:
                tqdm.write(f"  [FAIL] {sid}/{mod_name}: {e}")
                # print(f"[ERROR] Full traceback for {sid}/{mod_name}:", flush=True)
                import traceback
                traceback.print_exc()
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
