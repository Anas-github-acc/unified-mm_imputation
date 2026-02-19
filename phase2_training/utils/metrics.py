"""
PSNR and SSIM computation utilities for evaluation.

Uses scikit-image SSIM and standard PSNR formula.
Also provides PyTorch-based versions for use during training.
"""

import numpy as np
import torch
import torch.nn as nn


# ============================================================
#               NumPy-based (for final evaluation)
# ============================================================


def psnr_numpy(pred, gt, data_range=1.0):
    """
    Compute PSNR between two images.

    PSNR = 10 * log10(MAX^2 / MSE)

    Args:
        pred: predicted image (H, W) or (C, H, W) float
        gt: ground truth image, same shape
        data_range: max pixel value (1.0 for [0,1] normalized)

    Returns:
        PSNR value in dB
    """
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10.0 * np.log10(data_range**2 / mse)


def ssim_numpy(pred, gt, data_range=1.0):
    """
    Compute SSIM between two 2D images using scikit-image.

    Args:
        pred: (H, W) predicted image
        gt: (H, W) ground truth image
        data_range: max pixel value

    Returns:
        SSIM value
    """
    from skimage.metrics import structural_similarity
    return structural_similarity(pred, gt, data_range=data_range)


def compute_metrics_batch(pred_batch, gt_batch, data_range=1.0):
    """
    Compute PSNR and SSIM for a batch of images.

    Args:
        pred_batch: (B, C, H, W) numpy array
        gt_batch: (B, C, H, W) numpy array
        data_range: max pixel value

    Returns:
        dict with lists of per-sample PSNR and SSIM values
    """
    B, C, H, W = pred_batch.shape
    psnr_vals = []
    ssim_vals = []

    for b in range(B):
        p_list = []
        s_list = []
        for c in range(C):
            p_list.append(psnr_numpy(pred_batch[b, c], gt_batch[b, c], data_range))
            s_list.append(ssim_numpy(pred_batch[b, c], gt_batch[b, c], data_range))
        psnr_vals.append(np.mean(p_list))
        ssim_vals.append(np.mean(s_list))

    return {"psnr": psnr_vals, "ssim": ssim_vals}


# ============================================================
#            PyTorch-based (for use during training)
# ============================================================


def psnr_torch(pred, gt, data_range=1.0):
    """
    Compute PSNR using PyTorch tensors.

    Args:
        pred: (B, C, H, W) or (B, H, W) tensor
        gt: same shape tensor
        data_range: max pixel value

    Returns:
        Scalar PSNR value (averaged over batch)
    """
    mse = torch.mean((pred.float() - gt.float()) ** 2)
    if mse.item() < 1e-10:
        return torch.tensor(100.0)
    return 10.0 * torch.log10(torch.tensor(data_range**2) / mse)


def ssim_torch(pred, gt, window_size=11, data_range=1.0):
    """
    Simple SSIM implementation in PyTorch.

    Computes SSIM using a Gaussian window.

    Args:
        pred: (B, 1, H, W) tensor
        gt: (B, 1, H, W) tensor
        window_size: Gaussian window size
        data_range: max pixel value

    Returns:
        Scalar SSIM value
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Create Gaussian window
    sigma = 1.5
    gauss = torch.Tensor(
        [np.exp(-(x - window_size // 2) ** 2 / (2 * sigma**2)) for x in range(window_size)]
    )
    gauss = gauss / gauss.sum()

    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)

    window = _2D_window.expand(1, 1, window_size, window_size).contiguous()
    window = window.to(pred.device).type(pred.dtype)

    pad = window_size // 2

    mu1 = torch.nn.functional.conv2d(pred, window, padding=pad, groups=1)
    mu2 = torch.nn.functional.conv2d(gt, window, padding=pad, groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.nn.functional.conv2d(pred * pred, window, padding=pad, groups=1) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(gt * gt, window, padding=pad, groups=1) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(pred * gt, window, padding=pad, groups=1) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_map.mean()
