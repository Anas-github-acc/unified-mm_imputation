#!/usr/bin/env python3
"""
Step 0: Download IXI Dataset (T1, T2, PD)
Downloads and extracts IXI MRI scans from the official source.
Run: python download_ixi.py --output_dir ../data/raw/ixi
"""

import os
import sys
import argparse
import subprocess
import tarfile
from pathlib import Path


# IXI download URLs
IXI_URLS = {
    "T1": "https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar",
    "T2": "https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T2.tar",
    "PD": "https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-PD.tar",
}


def is_valid_tar(tar_path: str) -> bool:
    """Check if tar file is valid and complete."""
    try:
        with tarfile.open(tar_path, "r") as tar:
            # Try to list members to verify integrity
            tar.getmembers()
        return True
    except (tarfile.ReadError, EOFError, OSError):
        return False


def download_file(url: str, output_path: str) -> bool:
    """Download a file using wget or curl."""
    if os.path.exists(output_path):
        # Check if the tar file is valid
        if is_valid_tar(output_path):
            print(f"  [SKIP] {output_path} already exists and is valid")
            return True
        else:
            print(f"  [WARNING] {output_path} exists but is corrupted/incomplete")
            print(f"  [CLEANUP] Removing corrupted file")
            os.remove(output_path)

    print(f"  [DOWNLOAD] {url}")
    print(f"  [TO] {output_path}")

    # Try wget first, fall back to curl
    try:
        subprocess.run(
            ["wget", "-c", "-O", output_path, url],
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        try:
            subprocess.run(
                ["curl", "-L", "-C", "-", "-o", output_path, url],
                check=True,
            )
            return True
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            print(f"  [ERROR] Failed to download: {e}")
            return False


def extract_tar(tar_path: str, extract_dir: str):
    """Extract tar archive."""
    print(f"  [EXTRACT] {tar_path} -> {extract_dir}")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=extract_dir)
    print(f"  [DONE] Extracted to {extract_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download IXI dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data/raw/ixi",
        help="Directory to store downloaded IXI data",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["T1", "T2", "PD"],
        choices=["T1", "T2", "PD"],
        help="Which modalities to download",
    )
    parser.add_argument(
        "--keep_tar",
        action="store_true",
        help="Keep .tar files after extraction",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("IXI Dataset Downloader")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Modalities: {args.modalities}")
    print()

    for modality in args.modalities:
        url = IXI_URLS[modality]
        tar_path = output_dir / f"IXI-{modality}.tar"
        extract_dir = output_dir

        print(f"\n--- {modality} ---")
        
        # Check if files are already extracted
        pattern = f"IXI*-{modality}-*.nii*"
        existing_files = list(output_dir.glob(pattern))
        
        if existing_files and not tar_path.exists():
            print(f"  [SKIP] {modality} files already extracted ({len(existing_files)} files found)")
            continue
        
        success = download_file(url, str(tar_path))

        if success and tar_path.exists():
            extract_tar(str(tar_path), str(extract_dir))

            if not args.keep_tar:
                print(f"  [CLEANUP] Removing {tar_path}")
                os.remove(tar_path)

    # Count downloaded files per modality
    print("\n" + "=" * 60)
    print("Download Summary:")
    for modality in args.modalities:
        pattern = f"IXI*-{modality}-*.nii.gz"
        files = list(output_dir.glob(pattern))
        # Also check without .gz
        files += list(output_dir.glob(f"IXI*-{modality}-*.nii"))
        print(f"  {modality}: {len(files)} files")
    print("=" * 60)

    print("\nNext step: Run 'python build_manifest.py' to match subjects.")


if __name__ == "__main__":
    main()
