"""Full RGB float metrics, with clipping only at the metric boundary."""

import math

import cv2
import numpy as np
import torch


@torch.no_grad()
def image_metrics(prediction, target):
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("Metrics require equally shaped BCHW tensors")
    if min(prediction.shape[-2:]) < 11:
        raise ValueError("SSIM requires image dimensions >= 11")
    rows = []
    gaussian = cv2.getGaussianKernel(11, 1.5)
    kernel = gaussian @ gaussian.T
    for left, right in zip(prediction.detach().clamp(0, 1).cpu(), target.detach().clamp(0, 1).cpu()):
        left = left.permute(1, 2, 0).numpy().astype(np.float64)
        right = right.permute(1, 2, 0).numpy().astype(np.float64)
        filt = lambda value: cv2.filter2D(value, -1, kernel)[5:-5, 5:-5]
        ml, mr = filt(left), filt(right)
        vl, vr = filt(left * left) - ml * ml, filt(right * right) - mr * mr
        covariance = filt(left * right) - ml * mr
        ssim = ((2 * ml * mr + 0.01 ** 2) * (2 * covariance + 0.03 ** 2)) / (
            (ml * ml + mr * mr + 0.01 ** 2) * (vl + vr + 0.03 ** 2))
        mse = float(np.square(left - right).mean())
        rows.append({"psnr": -10 * math.log10(max(mse, 1e-20)), "ssim": float(ssim.mean())})
    return rows
