# Missing MRI Modality Imputation using MM-GAN

**Adapted for IXI Brain MRI Dataset (T1, T2, PD) with N4 Bias Field Correction Optimization**

---

This project implements a two-phase system for synthesizing missing MRI modalities using a Multi-Modal Generative Adversarial Network (MM-GAN). It is based on the [MM-GAN approach by Sharma et al.](https://github.com/trane293/mm-gan), originally designed for 4 modalities on the BRATS dataset (T1, T1c, T2, FLAIR), and adapted here to work with 3 modalities from the [IXI dataset](https://brain-development.org/ixi-dataset/) (T1, T2, PD).

The project includes a preprocessing optimization step -- N4 Bias Field Correction -- and provides a structured comparison between baseline and optimized pipelines using identical hyperparameters, data splits, and random seeds.

---

## Key Features

- **Multi-modal imputation**: Any subset of {T1, T2, PD} can be synthesized from any other available subset.
- **6 missing modality scenarios** via binary masking, covering all non-trivial combinations.
- **Curriculum learning**: Training gradually introduces harder imputation scenarios over time.
- **Implicit conditioning**: Real (known) channels are copied back into the generator output, ensuring the network only learns to fill in the missing modalities.
- **N4 Bias Field Correction** as a preprocessing optimization to reduce intensity inhomogeneity.
- **Resumable training** designed for 2-hour Kaggle GPU sessions with checkpoint saving and loading.
- **Comprehensive evaluation**: PSNR and SSIM computed per scenario and overall, with baseline vs. optimized comparison.

---

## Project Structure

```
missing_modality_imputation/
├── README.md
├── requirements.txt
├── phase1_preprocessing/
│   ├── configs/config.yaml
│   ├── run_phase1.py
│   └── scripts/
│       ├── download_ixi.py
│       ├── build_manifest.py
│       ├── register.py
│       ├── n4_correction.py
│       ├── extract_slices.py
│       └── qc_visualize.py
├── phase2_training/
│   ├── train.py
│   ├── evaluate.py
│   ├── compare_results.py
│   ├── kaggle_notebook.ipynb
│   ├── models/
│   │   ├── __init__.py
│   │   └── mmgan.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py
│       └── checkpoint.py
├── data/
│   ├── raw/ixi/
│   ├── processed/baseline/
│   ├── processed/optimized_n4/
│   └── manifests/
├── artifacts/qc/
└── docs/
    ├── phase1_preprocessing.md
    ├── phase2_kaggle_training.md
    └── results_and_video.md
```

---

## Two-Phase Workflow

### Phase 1: Preprocessing (Local CPU)

Phase 1 runs entirely on a local machine (no GPU required) and prepares the IXI dataset for training.

| Step | Description |
|------|-------------|
| 1. Download | Fetch IXI T1, T2, and PD `.tar` archives (~580 subjects). |
| 2. Build manifest | Match subjects possessing all 3 modalities; create train/val/test split (85% / 7.5% / 7.5%). |
| 3. Register | ANTsPy affine registration of T1 and T2 into PD space. |
| 4. N4 Bias Field Correction | Optimization step applied **after** registration to correct intensity inhomogeneity. Produces a second set of processed data (`optimized_n4/`) alongside the baseline. |
| 5. Extract 2D axial slices | Skip top/bottom 20 slices, center crop to 224x224, resize to 256x256, min-max normalize per volume, save as `.npy` files. |
| 6. QC visualization | Generate registration overlay images and N4 before/after comparison figures in `artifacts/qc/`. |

**Run Phase 1:**

```bash
python phase1_preprocessing/run_phase1.py
```

### Phase 2: Training (Kaggle GPU)

Phase 2 runs on Kaggle to leverage free GPU resources with a 2-hour session limit.

| Step | Description |
|------|-------------|
| 1. Upload data | Upload processed slices (baseline and/or optimized) to Kaggle as a dataset. |
| 2. Train baseline | Run `kaggle_notebook.ipynb` with `EXPERIMENT_NAME="baseline"` pointing to baseline data. |
| 3. Train optimized | Run `kaggle_notebook.ipynb` with `EXPERIMENT_NAME="optimized"` pointing to N4-corrected data. |
| 4. Resume training | Each Kaggle session has a 2-hour limit. Checkpoint resume handles multi-session training automatically. |
| 5. Evaluate and compare | After training both models, run evaluation and the comparison script to produce side-by-side metrics. |

**Run Phase 2:**

Open `phase2_training/kaggle_notebook.ipynb` on Kaggle and follow the instructions within.

---

## Model Architecture

### Generator

UNet architecture with:

- 8 encoder blocks and 7 decoder blocks
- Skip connections between corresponding encoder and decoder layers
- InstanceNorm for normalization
- Dropout (p=0.2) in deeper layers for regularization
- ReLU final activation

### Discriminator

PatchGAN architecture with:

- 4 downsampling convolutional blocks
- Outputs a grid of patch-level real/fake predictions

### Loss Function

LSGAN (Least Squares GAN) formulation:

- **Adversarial loss**: MSE-based (LSGAN)
- **Reconstruction loss**: L1 pixel-wise loss
- **Total generator loss**: `G_total = (1 - lambda) * L_GAN + lambda * L1`, where `lambda = 0.9`

### Missing Modality Handling

- 6 binary masking scenarios covering all non-trivial combinations of missing and available modalities among {T1, T2, PD}.
- Missing channels are filled with zeros before being passed to the generator.
- **Implicit conditioning**: After the generator produces its output, real (known) channels are copied back into the output tensor, so the network is only penalized for synthesizing the missing channels.

---

## Optimization: N4 Bias Field Correction

MRI images suffer from intensity inhomogeneity (bias field) caused by scanner imperfections and patient anatomy. This spatially smooth, low-frequency artifact distorts intensity values and can degrade the performance of downstream models.

**N4 Bias Field Correction** (N4ITK) estimates and removes this artifact, producing more uniform intensity distributions across each volume.

Key details:

- Applied **after** registration and **before** slice extraction.
- This is the sole differentiator between the baseline and optimized pipelines.
- Both experiments use identical hyperparameters, train/val/test splits, and random seeds, ensuring a fair and controlled comparison.

---

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies include PyTorch, ANTsPy, SimpleITK, nibabel, NumPy, scikit-image, and matplotlib. See `requirements.txt` for the full list.

---

## Quick Start

**Phase 1 (local preprocessing):**

```bash
python phase1_preprocessing/run_phase1.py
```

**Phase 2 (Kaggle training):**

Open `phase2_training/kaggle_notebook.ipynb` on Kaggle with a GPU accelerator enabled.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **PSNR** | Peak Signal-to-Noise Ratio (dB). Higher is better. Measures pixel-level reconstruction fidelity. |
| **SSIM** | Structural Similarity Index (0 to 1). Higher is better. Measures perceptual structural similarity. |

Metrics are computed:

- Per missing modality scenario (6 scenarios)
- Aggregated overall (mean across scenarios)
- For both baseline and optimized models, enabling direct comparison

---

## References

1. **MM-GAN**: Sharma R., et al., "Missing MRI Pulse Sequence Synthesis using Multi-Modal Generative Adversarial Network." IEEE Transactions on Medical Imaging, 2020. [GitHub](https://github.com/trane293/mm-gan)

2. **IXI Dataset**: Information eXtraction from Images. [https://brain-development.org/ixi-dataset/](https://brain-development.org/ixi-dataset/)

3. **N4ITK**: Tustison N.J., et al., "N4ITK: Improved N3 Bias Correction." IEEE Transactions on Medical Imaging, 2010.

---

## License

This project is intended for academic and research use only.
