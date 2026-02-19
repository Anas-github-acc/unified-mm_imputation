# Phase 1 - Preprocessing Pipeline

## 1. Overview

Phase 1 runs on local CPU and prepares the IXI brain MRI dataset for MM-GAN training. It produces two parallel data branches:

- **Baseline**: Registered slices without N4 bias field correction.
- **Optimized**: Registered slices with N4 bias field correction applied.

Both branches go through identical slice extraction after their respective preprocessing. The master runner for the entire phase is `run_phase1.py`, which orchestrates all steps in sequence.

---

## 2. Step-by-Step Pipeline

### Step 1: Download IXI Data (`download_ixi.py`)

Downloads the IXI brain MRI dataset from the Brain Development project hosted at Imperial College London.

- **Source**: `https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/`
- **Archives**: T1, T2, and PD-weighted tar archives.
- **Subjects**: Approximately 580 subjects acquired across 3 scanner sites:
  - Guys Hospital (1.5T)
  - Hammersmith Hospital (HH, 3T)
  - Institute of Psychiatry (IOP, 1.5T)
- **File format**: NIfTI compressed volumes following the naming convention `IXI{ID}-{Site}-{ScanID}-{Modality}.nii.gz`.
- **Output directory**: `data/raw/ixi/`

### Step 2: Build Manifest (`build_manifest.py`)

Constructs a structured manifest CSV that serves as the single source of truth for all downstream steps.

- Scans the raw directory for all NIfTI files.
- Parses each filename to extract: subject ID, site, scan ID, and modality.
- Filters to subjects that have **all 3 modalities** (T1, T2, PD). Subjects missing any modality are excluded.
- Splits subjects into train (85%), validation (7.5%), and test (7.5%) using a fixed random seed of 42 for reproducibility.
- **Output**: `data/manifests/manifest.csv`
- **Columns**: `subject_id`, `site`, `t1_path`, `t2_path`, `pd_path`, `split`

### Step 3: Registration (`register.py`)

Aligns all modalities into a common coordinate space so that voxel locations correspond across T1, T2, and PD volumes for each subject.

- **Library**: ANTsPy
- **Transform type**: Affine
- **Fixed image**: PD (used as the reference space)
- **Moving images**: T1 and T2
- **Operations per subject**: T1 -> PD registration, T2 -> PD registration
- **Output**: Registered NIfTI files saved to an intermediate directory.

**Why Affine registration?**
Affine transforms provide a good balance of accuracy versus computational speed for intra-subject brain MRI alignment. An affine transform handles translation, rotation, scaling, and shearing -- sufficient for correcting differences in head positioning and minor geometric distortions between modality acquisitions on the same subject. Non-linear (diffeomorphic) registration would add significant computation time with marginal benefit for same-subject, same-session scans.

### Step 4: N4 Bias Field Correction (`n4_correction.py`)

Corrects intensity inhomogeneity (bias field) caused by imperfections in MRI radio-frequency coils and other scanner-related factors. This is the key preprocessing difference between the baseline and optimized branches.

- **Applied to**: The optimized branch **only**. The baseline branch skips this step entirely.
- **Library**: ANTsPy's `n4_bias_field_correction` function.
- **Parameters**:
  - `shrink_factor=4` (downsamples internally for speed)
  - `convergence iterations=[50, 50, 50, 50]` (4-level multi-resolution)
- **Scope**: Applied to all 3 modalities (T1, T2, PD) **after** registration.
- **Purpose**: The bias field introduces smooth, low-frequency intensity variations across the image that can degrade downstream analysis and model performance. N4 estimates and removes this multiplicative field, producing more uniform tissue intensities.

This step is the core optimization under study: comparing MM-GAN performance when trained on bias-corrected versus uncorrected data.

### Step 5: Slice Extraction (`extract_slices.py`)

Converts 3D NIfTI volumes into 2D NumPy arrays suitable for 2D network training. This step runs identically for **both** the baseline and optimized branches.

- **Orientation**: Axial slices extracted along axis 2.
- **Slice filtering**:
  - Skips the top 20 and bottom 20 slices (predominantly air, skull boundaries, and regions with minimal brain content).
  - Skips any slice where less than 5% of voxels are non-zero.
- **Spatial processing**: Center-crops each slice to **224 x 224** pixels.
- **Intensity normalization**: Min-max normalization applied per volume, scaling intensities to the range [0, 1].
- **Channel stacking**: T1, T2, and PD slices are stacked into a single array of shape `(3, 224, 224)`.
- **Output format**: `.npy` files organized by branch and split:

```
data/processed/
  baseline/
    train/subXXX_sliceYYY.npy
    val/subXXX_sliceYYY.npy
    test/subXXX_sliceYYY.npy
  optimized_n4/
    train/subXXX_sliceYYY.npy
    val/subXXX_sliceYYY.npy
    test/subXXX_sliceYYY.npy
```

### Step 6: QC Visualization (`qc_visualize.py`)

Generates quality-control images for manual inspection of preprocessing results.

- **Registration QC**: Overlays registered T1 and T2 volumes on the PD reference to verify spatial alignment. Misalignment appears as color fringing or ghosting in the overlay.
- **N4 QC**: Side-by-side comparison of the same slice before and after N4 bias field correction, allowing visual confirmation that the bias field was removed without introducing artifacts.
- **Output directory**: `artifacts/qc/`

---

## 3. Configuration

All pipeline parameters are centralized in `configs/config.yaml`. Key settings include:

| Section        | Parameter                | Description                                                |
|----------------|--------------------------|------------------------------------------------------------|
| **paths**      | `raw_dir`                | Location of downloaded IXI archives (`data/raw/ixi/`)     |
|                | `manifest_path`          | Path to the output manifest CSV                            |
|                | `processed_dir`          | Root directory for processed slices                        |
|                | `qc_dir`                 | Directory for QC visualization outputs                     |
| **registration** | `type`                 | Registration transform type (e.g., `Affine`)               |
|                | `fixed_modality`         | Reference modality for registration (e.g., `PD`)          |
| **n4**         | `shrink_factor`          | Downsampling factor for N4 (default: 4)                    |
|                | `convergence_iterations` | Multi-resolution iteration schedule (default: [50,50,50,50]) |
| **slicing**    | `axis`                   | Slice extraction axis (default: 2, axial)                  |
|                | `skip_top`               | Number of top slices to skip (default: 20)                 |
|                | `skip_bottom`            | Number of bottom slices to skip (default: 20)              |
|                | `min_nonzero_frac`       | Minimum non-zero voxel fraction to keep a slice (default: 0.05) |
|                | `crop_size`              | Center-crop dimensions (default: 224)                      |
| **split**      | `train_ratio`            | Training set proportion (default: 0.85)                    |
|                | `val_ratio`              | Validation set proportion (default: 0.075)                 |
|                | `test_ratio`             | Test set proportion (default: 0.075)                       |
|                | `seed`                   | Random seed for reproducible splits (default: 42)          |
| **qc**         | `num_samples`            | Number of subjects to visualize for QC                     |
|                | `output_format`          | Image format for QC outputs (default: PNG)                 |

---

## 4. Running Phase 1

### Full pipeline

Run all steps in sequence via the master runner:

```bash
python phase1_preprocessing/run_phase1.py
```

### Individual steps

Each step can also be executed independently. The master runner supports skip flags to re-run specific stages without repeating earlier ones.

```bash
# Step 1: Download raw data
python phase1_preprocessing/scripts/download_ixi.py

# Step 2: Build manifest
python phase1_preprocessing/scripts/build_manifest.py

# Step 3: Registration
python phase1_preprocessing/scripts/register.py

# Step 4: N4 bias field correction (optimized branch only)
python phase1_preprocessing/scripts/n4_correction.py

# Step 5: Slice extraction (both branches)
python phase1_preprocessing/scripts/extract_slices.py

# Step 6: QC visualization
python phase1_preprocessing/scripts/qc_visualize.py
```

---

## 5. Expected Output

| Metric                        | Approximate Value              |
|-------------------------------|--------------------------------|
| Total IXI subjects downloaded | ~580                           |
| Subjects with all 3 modalities| ~500+                          |
| Valid slices per subject       | ~30-50                         |
| Total training slices (per branch) | ~15,000-25,000            |
| Disk size per branch          | ~2-3 GB                        |
| Total disk (raw + both branches) | ~20 GB                      |

Output directory structure after a successful run:

```
data/
  raw/ixi/                          # Downloaded NIfTI archives
  manifests/manifest.csv            # Subject manifest with splits
  processed/
    baseline/                       # Without N4 correction
      train/  val/  test/           # .npy slice files
    optimized_n4/                   # With N4 correction
      train/  val/  test/           # .npy slice files
artifacts/
  qc/                               # QC visualization PNGs
```

---

## 6. Troubleshooting

### ANTsPy installation

ANTsPy can be difficult to install via pip on some platforms due to its C++ compilation requirements. If `pip install antspyx` fails:

- Use conda: `conda install -c aramislab antspyx`
- Ensure you are using a compatible Python version (3.8-3.11 are generally well-supported).
- On Linux, verify that `cmake` and a C++ compiler (gcc/g++) are available.

### Memory usage

Registration is the most memory-intensive step. ANTsPy loads full 3D volumes into memory and computes spatial transforms.

- Process one subject at a time (the pipeline does this by default).
- Expect ~2-4 GB of RAM usage per subject during registration.
- If running on a machine with limited RAM, close other applications during this step.

### Disk space

Plan for approximately 20 GB of total disk space:

- Raw IXI data (3 tar archives): ~8-10 GB
- Intermediate registered volumes: ~3-5 GB
- Processed slices (baseline + optimized): ~4-6 GB
- QC artifacts: negligible

If disk space is constrained, delete intermediate registered volumes after slice extraction completes. The `.npy` slice files are the only artifacts required by Phase 2.

### Common errors

| Error | Likely cause | Fix |
|-------|-------------|-----|
| `FileNotFoundError` on raw data | Download incomplete or path mismatch | Re-run `download_ixi.py`; verify `config.yaml` paths |
| Registration produces blank images | Corrupt NIfTI file or extreme orientation mismatch | Check the source file manually with a viewer (e.g., ITK-SNAP); exclude the subject |
| N4 correction hangs | Very large volume with `shrink_factor=1` | Increase `shrink_factor` to 4 (default) or higher |
| Slice extraction yields 0 slices | Volume has unexpected orientation or is too small | Verify the volume dimensions; adjust `skip_top`/`skip_bottom` |
