"""OpenCV source-image reading and RGB float conversion."""

import cv2
import numpy as np
import torch


def imread(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def to_float_tensor(image):
    rgb = np.ascontiguousarray(image[..., ::-1])
    tensor = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    if rgb.dtype == np.uint8:
        tensor.div_(255.0)
    return tensor
