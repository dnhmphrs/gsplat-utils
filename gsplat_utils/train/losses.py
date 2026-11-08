"""Image losses in plain torch (no fused-ssim / torchmetrics dependency)."""
from __future__ import annotations


def gaussian_window(window_size: int, sigma: float, device, dtype):
    import torch

    x = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(x ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    return g[:, None] * g[None, :]  # (ws, ws)


def ssim(img1, img2, window_size: int = 11, sigma: float = 1.5, data_range: float = 1.0,
         padding: str = "same"):
    """Mean SSIM of two (B,C,H,W) tensors (Wang et al. 2004; same constants as 3DGS).

    padding='same' reproduces the original 3DGS loss; 'valid' matches fused-ssim.
    """
    import torch.nn.functional as F

    if img1.dim() == 3:
        img1, img2 = img1[None], img2[None]
    C = img1.shape[1]
    w = gaussian_window(window_size, sigma, img1.device, img1.dtype)
    w = w.expand(C, 1, window_size, window_size).contiguous()
    pad = window_size // 2 if padding == "same" else 0
    mu1 = F.conv2d(img1, w, padding=pad, groups=C)
    mu2 = F.conv2d(img2, w, padding=pad, groups=C)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, w, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, w, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, w, padding=pad, groups=C) - mu1_mu2
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def l1(a, b):
    return (a - b).abs().mean()


def psnr(a, b, data_range: float = 1.0):
    import torch

    mse = ((a - b) ** 2).mean()
    return 10.0 * torch.log10(data_range ** 2 / mse.clamp_min(1e-12))
