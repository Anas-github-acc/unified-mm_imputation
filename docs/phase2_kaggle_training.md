# Phase 2 - Kaggle GPU Training Guide

## 1. Overview

Phase 2 trains the MM-GAN model on Kaggle's free GPU tier (Tesla T4/P100, 2-hour sessions). Training is fully resumable via checkpoint management. Two identical training runs are performed:

- **Baseline**: trained on registered-only slices
- **Optimized**: trained on registered + N4-corrected slices

Both experiments use the same architecture, hyperparameters, and training procedure. The only difference is the input data preprocessing.

---

## 2. Preparing Data for Kaggle

### Uploading Slices as a Kaggle Dataset

After Phase 1 preprocessing is complete, package the processed slices and upload them as Kaggle datasets.

Create two Kaggle datasets:

1. `ixi-slices-baseline`: upload `data/processed/baseline/` (contains `train/`, `val/`, `test/` subdirectories)
2. `ixi-slices-optimized`: upload `data/processed/optimized_n4/`

Each dataset contains `.npy` files organized as:

```
{split}/{subject}_{slice}.npy
```

**Upload via Kaggle UI:**

Use Kaggle's "New Dataset" -> "Upload" feature to upload each directory.

**Upload via Kaggle API:**

```bash
kaggle datasets create -p ./data/processed/baseline/
kaggle datasets create -p ./data/processed/optimized_n4/
```

---

## 3. Using the Kaggle Notebook

### First Session (Baseline)

1. Upload `kaggle_notebook.ipynb` to Kaggle.
2. Enable GPU: Settings -> Accelerator -> GPU T4 x2 (or P100).
3. Add input dataset: `ixi-slices-baseline`.
4. Set the configuration cell:
   ```python
   EXPERIMENT_NAME = "baseline"
   DATA_PATH = "/kaggle/input/ixi-slices-baseline"
   ```
5. Run All cells.
6. Training will run for up to 1h50m (10-minute safety margin before session timeout).
7. Download the artifacts zip before the session ends.

### Resuming Training

1. Re-open the notebook.
2. Checkpoint files are stored in `/kaggle/working/checkpoints/`, which persists between "Save & Run All" runs but **not** between new sessions.
3. To resume across sessions: upload the artifacts zip as a new Kaggle dataset, or use Kaggle's output mechanism to reference prior run outputs.
4. The notebook auto-detects existing checkpoints and resumes from the latest one.

### Second Experiment (Optimized)

1. Change the configuration cell:
   ```python
   EXPERIMENT_NAME = "optimized"
   DATA_PATH = "/kaggle/input/ixi-slices-optimized"
   ```
2. Run All cells.
3. Follow the same checkpoint/resume process as the baseline experiment.

---

## 4. Training Details

### Architecture

- **Generator**: UNet with 8 encoder blocks and 7 decoder blocks, skip connections, InstanceNorm, and ReLU output activation.
- **Discriminator**: PatchGAN with 4 downsampling blocks, outputting 16x16 patch predictions for 256x256 input images.
- Both networks are initialized with a normal distribution (mean=0, std=0.02).

### Hyperparameters (Identical for Both Experiments)

| Parameter      | Value                    |
|----------------|--------------------------|
| Epochs         | 60                       |
| Batch size     | 8                        |
| LR (G and D)   | 2e-4                     |
| Adam betas     | (0.5, 0.999)             |
| Lambda pixel   | 0.9                      |
| LR scheduler   | StepLR, step=20, gamma=0.5 |
| Impute type    | zeros                    |
| Seed           | 42                       |
| Image size     | 256x256                  |

### Loss Functions

- **Adversarial (LSGAN)**: MSELoss between discriminator output and target labels.
- **Pixel reconstruction**: L1Loss between generated and real images.
- **Combined generator loss**:
  ```
  G_total = (1 - 0.9) * L_GAN + 0.9 * L1 = 0.1 * L_GAN + 0.9 * L1
  ```

### Missing Modality Scenarios (6 Total for 3 Modalities)

| Scenario | T1 | T2 | PD | Missing     |
|----------|----|----|-----|-------------|
| 011      | 0  | 1  | 1   | T1          |
| 101      | 1  | 0  | 1   | T2          |
| 110      | 1  | 1  | 0   | PD          |
| 001      | 0  | 0  | 1   | T1, T2      |
| 010      | 0  | 1  | 0   | T1, PD      |
| 100      | 1  | 0  | 0   | T2, PD      |

### Curriculum Learning

Training is divided into three stages based on epoch progression (N = total epochs):

- **Epochs 0 to N/3**: Only scenarios with 1 missing modality (011, 101, 110).
- **Epochs N/3 to 2N/3**: Add 2-missing scenarios except the hardest (011, 101, 110, 001, 010).
- **Epochs 2N/3 to N**: All 6 scenarios.

### Implicit Conditioning

After the generator produces its output, real channels (the available modalities) are copied back into the generated image. This ensures the network focuses exclusively on synthesizing the missing channels rather than reconstructing already-available information.

---

## 5. Checkpoint System

### What Gets Saved

Each checkpoint file contains:

- Generator state dict
- Discriminator state dict
- Optimizer G state dict
- Optimizer D state dict
- Scheduler G state dict
- Scheduler D state dict
- Current epoch
- Best PSNR and SSIM values
- Training history (loss and metric lists)

### File Naming

- Regular checkpoints: `mmgan_epoch{N}.pth`
- Best model (by validation PSNR): `mmgan_best.pth`
- Only the last 3 regular checkpoints are retained to conserve disk space.

### Resume Logic

1. Scans the checkpoint directory for the latest checkpoint file.
2. Loads all state dicts (model weights, optimizers, schedulers).
3. Continues training from the next epoch.
4. Preserves optimizer momentum, scheduler state, and RNG seeds.

---

## 6. Evaluation

Run the evaluation cell only after all 60 epochs are complete for a given experiment:

- Computes PSNR and SSIM on the test set for all 6 missing modality scenarios.
- Generates visual comparison images (input with missing channels, generated output, ground truth).
- Saves `metrics.json` for later comparison between experiments.

---

## 7. Comparison

After both experiments (baseline and optimized) are evaluated:

- Run `compare_results.py` locally with both `metrics.json` files.
- Alternatively, use the comparison code embedded in the notebook.
- Outputs:
  - Comparison table (per-scenario PSNR and SSIM for both experiments)
  - Bar charts (PSNR and SSIM per scenario, side-by-side)
  - Combined results JSON

---

## 8. Time Estimates

| Metric                          | Estimate            |
|---------------------------------|---------------------|
| Training slices                 | ~20K                |
| Batches per epoch (batch_size=8)| ~2500               |
| Time per epoch (T4 GPU)         | ~3-5 minutes        |
| Total for 60 epochs             | ~3-5 hours          |
| Kaggle sessions per experiment  | 2-3                 |
| Total sessions (both experiments)| 4-6                |

---

## 9. Tips

- Always download checkpoints before the Kaggle session ends.
- Monitor training with TensorBoard:
  ```python
  %load_ext tensorboard
  %tensorboard --logdir ./logs
  ```
- If training diverges (NaN losses), reduce the learning rate or verify data integrity.
- Keep batch size at 8 for T4 (16GB VRAM). Reduce to 4 if encountering out-of-memory errors.
