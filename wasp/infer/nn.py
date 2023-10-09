from typing import Optional

import cv2
import numpy as np


def nninput(
    image: np.ndarray,
    mean: float = 127.5,
    std: float = 128.0,
    shape: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    if shape is None:
        *shape, _ = image.shape
    return cv2.dnn.blobFromImage(
        image,
        1.0 / std,
        shape,
        (mean, mean, mean),
        swapRB=True,
    )


def nnoutput(blob: np.ndarray, norm=255) -> np.ndarray:
    # Convert to channels last as normal image
    channels_last = blob.transpose((0, 2, 3, 1))[0]

    # Convert RGB image
    rgb = np.clip(norm * channels_last, 0, 255).astype(np.uint8)

    # Convert back to BGR
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
