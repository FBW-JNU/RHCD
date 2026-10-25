"""RGB source crops and lossless float carrier files."""

from pathlib import Path
import hashlib

import cv2
import numpy as np
import torch

from dataset.div2k import imread, to_float_tensor
from dataset.transform import center_crop_or_resize


def read_source(path, crop_size):
    if int(crop_size) < 1:
        raise ValueError("Source crop size must be positive")
    image = center_crop_or_resize(imread(str(path)), int(crop_size))
    return to_float_tensor(np.ascontiguousarray(image)).unsqueeze(0)


def source_crop_sha256(path, crop_size):
    image = center_crop_or_resize(imread(str(path)), int(crop_size))
    rgb = np.ascontiguousarray(image[..., ::-1])
    return hashlib.sha256(rgb.tobytes()).hexdigest()


def read_carrier(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
        if array.dtype != np.float32:
            raise ValueError("NPY carriers must be RGB HWC float32")
    else:
        array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if array is None:
            raise FileNotFoundError(path)
        if array.ndim != 3 or array.shape[2] != 3 or array.dtype not in (np.uint8, np.uint16):
            raise ValueError("Image carriers must be three-channel uint8 or uint16")
        peak = np.iinfo(array.dtype).max
        array = np.ascontiguousarray(array[..., ::-1]).astype(np.float32) / peak
    if array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) < 2 or not np.isfinite(array).all():
        raise ValueError("Expected a finite HWC RGB carrier")
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).unsqueeze(0)


def save_tensor(directory, name, tensor):
    directory = Path(directory)
    array = tensor.detach().cpu().float().squeeze(0).permute(1, 2, 0).contiguous().numpy()
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite output: " + name)
    np.save(directory / (name + ".npy"), array, allow_pickle=False)
    preview = np.rint(array.clip(0, 1) * 255).astype(np.uint8)
    if not cv2.imwrite(str(directory / (name + ".png")), np.ascontiguousarray(preview[..., ::-1])):
        raise OSError("Failed to save preview: " + name)
    return {"shape_hwc": list(array.shape), "min": float(array.min()), "max": float(array.max()),
            "raw_file": name + ".npy", "preview_file": name + ".png"}
