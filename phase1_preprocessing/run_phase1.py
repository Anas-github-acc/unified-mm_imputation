#!/usr/bin/env python3
"""manifest (assumes data already downloaded)
  2. Register volumes (T1/T2 -> PD)
  3. N4 bias field correction
  4. Extract slices (baseline + optimized)
Phase 1 Master Runner
Runs the full preprocessing pipeline in sequence:
  1. Build m
  5. QC visualization

Run: python run_phase1.py --ixi_dir ../data/raw/ixi
"""

import os
import sys
import argparse
import time
import subprocess
from pathlib import Path


def run_step(script: str, args: list, step_name: str):
    """Run a preprocessing step as subprocess."""
    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print(f"{'='*60}")

    cmd = [sys.executable, script] + args
    print(f"Command: {' '.join(cmd)}")
    print()

    start = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

    elapsed = time.time() - start
    status = "SUCCESS" if result.returncode == 0 else "FAILED"
    print(f"\n[{status}] {step_name} ({elapsed:.1f}s)")

    if result.returncode != 0:
        print(f"ERROR: Step '{step_name}' failed with return code {result.returncode}")
        sys.exit(1)

    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Master Runner")
    parser.add_argument("--ixi_dir", type=str, default="../data/raw/ixi",
                        help="Directory containing raw IXI NIfTI files")
    parser.add_argument("--config", type=str, default="../phase1_preprocessing/configs/config.yaml")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download step (data already present)")
    parser.add_argument("--skip_registration", action="store_true",
                        help="Skip registration (already done)")
    parser.add_argument("--skip_n4", action="store_true",
                        help="Skip N4 correction (already done)")
    parser.add_argument("--skip_slicing", action="store_true",
                        help="Skip slice extraction")
    parser.add_argument("--skip_qc", action="store_true",
                        help="Skip QC visualization")
    parser.add_argument("--max_subjects", type=int, default=-1,
                        help="Process at most N subjects (-1=all)")
    args = parser.parse_args()

    scripts_dir = Path(__file__).parent / "scripts"

    print("=" * 60)
    print("PHASE 1: IXI Preprocessing Pipeline")
    print("=" * 60)
    print(f"IXI directory:  {args.ixi_dir}")
    print(f"Config:         {args.config}")
    print(f"Max subjects:   {'all' if args.max_subjects == -1 else args.max_subjects}")

    total_start = time.time()
    timings = {}

    subj_args = []
    if args.max_subjects > 0:
        subj_args = ["--max_subjects", str(args.max_subjects)]

    # Step 1: Build manifest
    timings["manifest"] = run_step(
        str(scripts_dir / "build_manifest.py"),
        ["--ixi_dir", args.ixi_dir, "--config", args.config],
        "Build Subject Manifest",
    )

    # Step 2: Registration
    if not args.skip_registration:
        timings["registration"] = run_step(
            str(scripts_dir / "register.py"),
            ["--manifest", "../data/manifests/subject_manifest.csv",
             "--config", args.config] + subj_args,
            "ANTsPy Registration (T1/T2 -> PD)",
        )

    # Step 3: N4 Bias Field Correction
    if not args.skip_n4:
        timings["n4"] = run_step(
            str(scripts_dir / "n4_correction.py"),
            ["--config", args.config] + subj_args,
            "N4 Bias Field Correction",
        )

    # Step 4: Slice Extraction
    if not args.skip_slicing:
        timings["slicing"] = run_step(
            str(scripts_dir / "extract_slices.py"),
            ["--mode", "both", "--config", args.config],
            "Slice Extraction (Baseline + Optimized)",
        )

    # Step 5: QC Visualization
    if not args.skip_qc:
        timings["qc"] = run_step(
            str(scripts_dir / "qc_visualize.py"),
            ["--max_subjects", "5"],
            "QC Visualization",
        )

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print("=" * 60)
    print("\nTimings:")
    for step, t in timings.items():
        print(f"  {step:20s}: {t:8.1f}s")
    print(f"  {'TOTAL':20s}: {total_elapsed:8.1f}s")

    print("\nOutputs:")
    print(f"  Manifest:     ../data/manifests/subject_manifest.csv")
    print(f"  Registered:   ../data/processed/registered/")
    print(f"  N4 Corrected: ../data/processed/n4_corrected/")
    print(f"  Baseline:     ../data/processed/baseline/")
    print(f"  Optimized:    ../data/processed/optimized_n4/")
    print(f"  QC:           ../artifacts/qc/")

    print("\n--- NEXT: Upload processed data to Kaggle for Phase 2 ---")
    print("=" * 60)


if __name__ == "__main__":
    main()
