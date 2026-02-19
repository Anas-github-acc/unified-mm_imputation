# Results Analysis and Video Presentation Guide

This document provides guidance for analyzing experiment results and preparing video presentation materials for the missing modality imputation project.

---

## 1. Expected Results Format

### Per-Scenario Metrics (metrics.json)

Each experiment produces a JSON file with the following structure:

```json
{
  "per_scenario": {
    "011": {
      "missing": ["T1"],
      "available": ["T2", "PD"],
      "psnr_mean": 25.5,
      "psnr_std": 3.2,
      "ssim_mean": 0.85,
      "ssim_std": 0.05,
      "n_samples": 1500
    },
    "101": {
      "missing": ["T2"],
      "available": ["T1", "PD"],
      "psnr_mean": 24.8,
      "psnr_std": 2.9,
      "ssim_mean": 0.83,
      "ssim_std": 0.06,
      "n_samples": 1500
    }
  },
  "overall": {
    "psnr_mean": 23.0,
    "ssim_mean": 0.82
  }
}
```

### Comparison Output

`compare_results.py` produces:

- `comparison_chart.png`: Side-by-side bar charts (PSNR and SSIM per scenario)
- `combined_results.json`: Both experiments' metrics plus improvement deltas
- Console table showing per-scenario and overall improvements

---

## 2. Results Table Template

| Scenario | Missing | Available | Baseline PSNR | Optimized PSNR | Delta | Baseline SSIM | Optimized SSIM | Delta |
|----------|---------|-----------|---------------|----------------|-------|---------------|----------------|-------|
| 011 | T1 | T2, PD | -- | -- | -- | -- | -- | -- |
| 101 | T2 | T1, PD | -- | -- | -- | -- | -- | -- |
| 110 | PD | T1, T2 | -- | -- | -- | -- | -- | -- |
| 001 | T1, T2 | PD | -- | -- | -- | -- | -- | -- |
| 010 | T1, PD | T2 | -- | -- | -- | -- | -- | -- |
| 100 | T2, PD | T1 | -- | -- | -- | -- | -- | -- |
| **Overall** | | | **--** | **--** | **--** | **--** | **--** | **--** |

Fill in after running both experiments.

---

## 3. Key Figures to Generate

1. **QC Registration Overlay** (`artifacts/qc/`): Shows registered T1/T2 overlaid on PD reference. Demonstrates alignment quality.
2. **N4 Before/After** (`artifacts/qc/`): Side-by-side showing intensity inhomogeneity correction.
3. **Training Loss Curves**: From TensorBoard logs. Generator and Discriminator loss over epochs.
4. **Validation PSNR/SSIM Curves**: Metric improvement over training.
5. **Per-Scenario Bar Chart** (`comparison_chart.png`): Baseline vs Optimized PSNR and SSIM.
6. **Visual Comparisons** (`results/{experiment}/visuals/`): Input | Ground Truth | Synthesized for select scenarios.

---

## 4. How to Generate Figures

After both experiments are complete, run the comparison script:

```bash
python phase2_training/compare_results.py \
    --baseline_results ./results/baseline/metrics.json \
    --optimized_results ./results/optimized/metrics.json \
    --output_dir ./results/comparison
```

For training loss curves, launch TensorBoard:

```bash
tensorboard --logdir ./logs
# Screenshot the relevant plots
```

---

## 5. Video Presentation Narration Guide

Suggested video structure (5-8 minutes):

### Slide 1: Introduction (30-60s)

- Problem: In clinical MRI, some modalities are often missing due to time, cost, and patient comfort constraints.
- Solution: Use GANs to synthesize missing modalities from available ones.
- Dataset: IXI brain MRI (T1, T2, PD) - approximately 580 subjects across 3 sites.

### Slide 2: Method Overview (60-90s)

- Based on MM-GAN architecture (Sharma et al.).
- Adapted from 4 modalities (BRATS) to 3 modalities (IXI).
- 2-phase pipeline: CPU preprocessing + GPU training.
- 6 missing modality scenarios.
- Key techniques: curriculum learning, implicit conditioning, LSGAN + L1 loss.

### Slide 3: Preprocessing Pipeline (60s)

- Show pipeline diagram: Download -> Register -> N4 -> Slice -> Train.
- Registration: ANTsPy Affine (T1/T2 to PD space).
- Show QC registration overlay image.
- N4 Bias Field Correction: show before/after comparison.

### Slide 4: Optimization - N4 Bias Field Correction (60s)

- What is bias field? Intensity inhomogeneity from scanner coil sensitivity.
- N4 algorithm corrects this, making intensity distributions more uniform.
- Hypothesis: Corrected data should lead to better imputation because the model learns true tissue contrast rather than scanner artifacts.
- Show N4 before/after QC image.

### Slide 5: Model Architecture (45-60s)

- Generator: UNet with skip connections.
- Discriminator: PatchGAN.
- Missing modality handling: binary scenarios, implicit conditioning.
- Show architecture diagram if available.

### Slide 6: Training Details (30-45s)

- 60 epochs, batch size 8, learning rate 2e-4 with StepLR decay.
- LSGAN + 0.9*L1 loss.
- Curriculum learning progression.
- Kaggle T4 GPU, approximately 3-5 hours per experiment.

### Slide 7: Results (90-120s)

- Show comparison table (baseline vs optimized).
- Show bar chart figure.
- Show visual comparison examples (best and worst cases).
- Highlight: which scenarios improved most with N4?
- Discuss: single-missing vs multi-missing performance.

### Slide 8: Conclusion (30-45s)

- N4 bias field correction as preprocessing optimization.
- Expected improvement in PSNR/SSIM (or discussion if improvement was marginal).
- Future work: more advanced registration (SyN), other optimizations, 3D models.
- Thank you.

---

## 6. Interpreting Results

### What Good Results Look Like

- PSNR > 25 dB generally indicates good reconstruction quality.
- SSIM > 0.85 indicates good structural preservation.
- Single-missing scenarios (011, 101, 110) should outperform multi-missing (001, 010, 100).
- N4 optimization should show consistent (even if small) improvements across scenarios.

### If N4 Shows No Improvement

- Possible if IXI data already has relatively uniform intensity (high-field scanners).
- The comparison is still valuable as a negative result -- it demonstrates the optimization was tested rigorously.
- Discussion point: bias field may be more impactful on lower-quality data or different datasets.

### Common Issues

- Very low PSNR in multi-missing scenarios is expected (harder task).
- Mode collapse: all outputs look similar. Check discriminator loss for signs of instability.
- Blurry outputs: increase GAN loss weight (decrease `lambda_pixel`).

---

## 7. File Checklist for Final Submission

- [ ] `results/baseline/metrics.json`
- [ ] `results/optimized/metrics.json`
- [ ] `results/comparison/comparison_chart.png`
- [ ] `results/comparison/combined_results.json`
- [ ] `results/baseline/visuals/` (comparison PNGs)
- [ ] `results/optimized/visuals/` (comparison PNGs)
- [ ] `artifacts/qc/` (registration and N4 QC images)
- [ ] TensorBoard screenshots (loss curves, metric curves)
- [ ] Completed results table (filled in from template above)
- [ ] Video recording
