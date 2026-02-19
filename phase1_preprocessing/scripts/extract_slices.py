#!/usr/bin/env python3
"""
Step 4: Slice Extraction
Extracts 2D axial slices from registered (and optionally N4-corrected) volumes.
Produces two separate datasets:
  - Baseline: from registered volumes (no N4)
  - Optimized: from N4-corrected volumes

Run: python extract_slices.py --mode both
"""

import os
import sys
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import nibabel as nib
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm


def load_volume_nibabel(nifti_path: str) -> np.ndarray:
    """Load a NIfTI volume and return as numpy array."""
    img = nib.load(nifti_path)
    data = img.get_fdata().astype(np.float32)
    return data


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Min-max normalize a volume to [0, 1]."""
    vmin = volume.min()
    vmax = volume.max()
    if vmax - vmin > 1e-8:
        return (volume - vmin) / (vmax - vmin)
    return np.zeros_like(volume)


def center_crop_2d(slice_2d: np.ndarray, crop_size: tuple) -> np.ndarray:
    """Center-crop a 2D slice to crop_size (H, W)."""
    h, w = slice_2d.shape
    ch, cw = crop_size

    # If slice is smaller than crop, pad with zeros
    if h < ch or w < cw:
        padded = np.zeros((max(h, ch), max(w, cw)), dtype=slice_2d.dtype)
        ph = (max(h, ch) - h) // 2
        pw = (max(w, cw) - w) // 2
        padded[ph:ph+h, pw:pw+w] = slice_2d
        slice_2d = padded
        h, w = slice_2d.shape

    start_h = (h - ch) // 2
    start_w = (w - cw) // 2
    return slice_2d[start_h:start_h+ch, start_w:start_w+cw]


def extract_subject_slices(
    subject_id: str,
    modality_paths: dict,
    output_dir: str,
    split: str,
    config: dict,
    save_qc: bool = True,
    qc_dir: str = None,
):
    """
    Extract 2D axial slices from a subject's registered volumes.
    Saves as .npy arrays and optional QC PNGs.
    """
    slice_cfg = config.get("slicing", {})
    axis = slice_cfg.get("axis", 2)
    skip_top = slice_cfg.get("skip_top", 20)
    skip_bottom = slice_cfg.get("skip_bottom", 20)
    crop_size = tuple(slice_cfg.get("crop_size", [224, 224]))
    do_normalize = slice_cfg.get("normalize", True)

    qc_cfg = config.get("qc", {})
    qc_samples = qc_cfg.get("samples_per_subject", 3)

    out_split = Path(output_dir) / split
    out_split.mkdir(parents=True, exist_ok=True)

    # Load all modalities
    volumes = {}
    for mod_name, mod_path in modality_paths.items():
        if not os.path.exists(mod_path):
            raise FileNotFoundError(f"Missing: {mod_path}")
        vol = load_volume_nibabel(mod_path)
        if do_normalize:
            vol = normalize_volume(vol)
        volumes[mod_name] = vol

    # Get number of slices along the chosen axis
    ref_vol = list(volumes.values())[0]
    n_slices = ref_vol.shape[axis]

    # Determine slice range (skip top/bottom)
    start_slice = skip_bottom
    end_slice = n_slices - skip_top

    if end_slice <= start_slice:
        raise ValueError(f"No valid slices for {subject_id}: n_slices={n_slices}, skip={skip_bottom}/{skip_top}")

    # QC slice indices
    qc_indices = np.linspace(start_slice, end_slice - 1, qc_samples, dtype=int).tolist() if save_qc else []

    slice_count = 0
    for s in range(start_slice, end_slice):
        # Extract slice from each modality
        slices = {}
        for mod_name, vol in volumes.items():
            if axis == 0:
                sl = vol[s, :, :]
            elif axis == 1:
                sl = vol[:, s, :]
            else:
                sl = vol[:, :, s]

            sl = center_crop_2d(sl, crop_size)
            slices[mod_name] = sl

        # Skip mostly-empty slices (< 5% non-zero)
        all_data = np.stack(list(slices.values()))
        if np.count_nonzero(all_data) / all_data.size < 0.05:
            continue

        # Stack modalities: (3, H, W) for T1, T2, PD
        stacked = np.stack([slices["T1"], slices["T2"], slices["PD"]], axis=0)

        # Save as .npy
        slice_name = f"{subject_id}_slice{s:04d}"
        npy_path = out_split / f"{slice_name}.npy"
        np.save(npy_path, stacked)
        slice_count += 1

        # Save QC PNG
        if save_qc and s in qc_indices and qc_dir:
            qc_path = Path(qc_dir)
            qc_path.mkdir(parents=True, exist_ok=True)

            # Create side-by-side image: T1 | T2 | PD
            h, w = crop_size
            canvas = np.zeros((h, w * 3), dtype=np.float32)
            canvas[:, 0:w] = slices["T1"]
            canvas[:, w:2*w] = slices["T2"]
            canvas[:, 2*w:3*w] = slices["PD"]

            canvas = (canvas * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(canvas, mode="L")
            img.save(qc_path / f"{slice_name}_qc.png")

    return slice_count


def process_single_subject(row_data, input_path, modality_files, output_dir, config, qc_dir, mode):
    """
    Worker function to process a single subject.
    Designed to be used with multiprocessing.Pool.
    """
    sid = row_data["subject_id"]
    split = row_data["split"]
    
    subj_dir = input_path / sid
    if not subj_dir.exists():
        return {"subject_id": sid, "split": split, "n_slices": 0, "status": "SKIP", "message": "directory not found"}
    
    mod_paths = {}
    for mod_name, filename in modality_files.items():
        mod_paths[mod_name] = str(subj_dir / filename)
    
    try:
        n_slices = extract_subject_slices(
            subject_id=sid,
            modality_paths=mod_paths,
            output_dir=output_dir,
            split=split,
            config=config,
            save_qc=(qc_dir is not None),
            qc_dir=qc_dir,
        )
        return {"subject_id": sid, "split": split, "n_slices": n_slices, "status": "OK", "message": ""}
    except Exception as e:
        return {"subject_id": sid, "split": split, "n_slices": 0, "status": "FAIL", "message": str(e)}


def process_all_subjects(
    input_dir: str,
    output_dir: str,
    manifest_path: str,
    config: dict,
    qc_dir: str = None,
    mode: str = "baseline",
    n_workers: int = None,
):
    """Process all subjects for slice extraction using multiprocessing."""
    df = pd.read_csv(manifest_path)

    # Determine modality file names based on mode
    modality_files = {
        "T1": "T1_registered.nii.gz",
        "T2": "T2_registered.nii.gz",
        "PD": "PD.nii.gz",
    }

    # Determine number of workers
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)  # Leave 1 core free
    
    print(f"\nMode: {mode}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Workers: {n_workers} / {cpu_count()} available cores")

    input_path = Path(input_dir)
    total_slices = 0
    results = []

    # Prepare subject data for parallel processing
    subject_data = []
    for _, row in df.iterrows():
        subject_data.append({
            "subject_id": row["subject_id"],
            "split": row["split"]
        })
    
    # Create partial function with fixed parameters
    worker_fn = partial(
        process_single_subject,
        input_path=input_path,
        modality_files=modality_files,
        output_dir=output_dir,
        config=config,
        qc_dir=qc_dir,
        mode=mode
    )
    
    # Process subjects in parallel
    with Pool(processes=n_workers) as pool:
        # Use imap for progress bar support
        for result in tqdm(
            pool.imap(worker_fn, subject_data),
            total=len(subject_data),
            desc=f"Slicing ({mode})"
        ):
            status = result["status"]
            sid = result["subject_id"]
            n_slices = result["n_slices"]
            
            if status == "OK":
                total_slices += n_slices
                results.append({"subject_id": sid, "split": result["split"], "n_slices": n_slices})
                tqdm.write(f"  [OK] {sid}: {n_slices} slices")
            elif status == "SKIP":
                tqdm.write(f"  [SKIP] {sid}: {result['message']}")
            else:  # FAIL
                tqdm.write(f"  [FAIL] {sid}: {result['message']}")

    # Save slice manifest
    slice_df = pd.DataFrame(results)
    out_path = Path(output_dir)
    slice_df.to_csv(out_path / "slice_manifest.csv", index=False)

    print(f"\n  Total slices extracted: {total_slices}")
    print(f"  Manifest: {out_path / 'slice_manifest.csv'}")

    return total_slices


def main():
    parser = argparse.ArgumentParser(description="Extract 2D axial slices from 3D volumes")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["baseline", "optimized", "both"],
                        help="Which dataset branch to process")
    parser.add_argument("--registered_dir", type=str, default="data/processed/registered")
    parser.add_argument("--n4_dir", type=str, default="data/processed/n4_corrected")
    parser.add_argument("--baseline_out", type=str, default="data/processed/baseline")
    parser.add_argument("--optimized_out", type=str, default="data/processed/optimized_n4")
    parser.add_argument("--manifest", type=str, default="data/manifests/subject_manifest.csv")
    parser.add_argument("--config", type=str, default="phase1_preprocessing/configs/config.yaml")
    parser.add_argument("--qc_dir", type=str, default="artifacts/qc")
    parser.add_argument("--workers", type=int, default=None,
                        help=f"Number of parallel workers (default: {max(1, cpu_count() - 1)} of {cpu_count()} cores)")
    args = parser.parse_args()

    # Load config
    config = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

    print("=" * 60)
    print("Slice Extraction Pipeline")
    print("=" * 60)

    if args.mode in ("baseline", "both"):
        print("\n--- BASELINE (No N4) ---")
        process_all_subjects(
            input_dir=args.registered_dir,
            output_dir=args.baseline_out,
            manifest_path=args.manifest,
            config=config,
            qc_dir=os.path.join(args.qc_dir, "baseline") if args.qc_dir else None,
            mode="baseline",
            n_workers=args.workers,
        )

    if args.mode in ("optimized", "both"):
        print("\n--- OPTIMIZED (N4 Corrected) ---")
        process_all_subjects(
            input_dir=args.n4_dir,
            output_dir=args.optimized_out,
            manifest_path=args.manifest,
            config=config,
            qc_dir=os.path.join(args.qc_dir, "optimized") if args.qc_dir else None,
            mode="optimized",
            n_workers=args.workers,
        )

    print("\n" + "=" * 60)
    print("Slice extraction complete!")
    print("Next step: Upload to Kaggle and run Phase 2 training.")
    print("=" * 60)


if __name__ == "__main__":
    main()
