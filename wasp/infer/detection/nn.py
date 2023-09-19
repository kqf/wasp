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
