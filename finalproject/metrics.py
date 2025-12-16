"""
Evaluation metrics for model inversion attacks and reconstructions.
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim


def compute_ssim(a, b):
    """
    Compute Structural Similarity Index (SSIM) between two images.

    SSIM measures the similarity between two images based on luminance,
    contrast, and structure. Higher values indicate more similar images.

    Args:
        a, b: Images in [0..1], shape HxW

    Returns:
        SSIM score (higher is better, max 1.0)
    """
    return ssim(a, b, data_range=1.0)


def compute_psnr(a, b):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two images.

    PSNR measures the ratio between the maximum possible power of a signal
    and the power of corrupting noise. Higher values indicate better quality.

    Args:
        a, b: Images in [0..1], shape HxW

    Returns:
        PSNR in dB (higher is better)
    """
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr
