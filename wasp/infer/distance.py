import cv2
import numpy as np
from skimage.transform import SimilarityTransform


def distance2bbox(
    points: np.ndarray,
    distance: np.ndarray,
    max_shape: tuple = None,
) -> np.ndarray:
    """Decode distance predictions to bounding boxes.

    Parameters:
        points (np.ndarray): of shape (n, 2), where each row is [x, y].
        distance (np.ndarray): from a point to 4 boundaries [l, t, r, b].
        max_shape (tuple): of the image.


    Returns:
        np.ndarray: Decoded bounding boxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]

    if max_shape is not None:
        x1 = np.clip(x1, a_min=0, a_max=max_shape[1])
        y1 = np.clip(y1, a_min=0, a_max=max_shape[0])
        x2 = np.clip(x2, a_min=0, a_max=max_shape[1])
        y2 = np.clip(y2, a_min=0, a_max=max_shape[0])

    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(
    points: np.ndarray,
    distance: np.ndarray,
    max_shape: tuple = None,
) -> np.ndarray:
    """Decode distance predictions to keypoints.

    Parameters:
        points (np.ndarray): of shape (n, 2), where each row is [x, y].
        distance (np.ndarray): from a point to 4 boundaries [l, t, r, b].
        max_shape (tuple): of the image.

    Returns:
        np.ndarray: Decoded keypoints.
    """
    preds: list[np.ndarray] = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]

        if max_shape is not None:
            px = np.clip(px, a_min=0, a_max=max_shape[1])
            py = np.clip(py, a_min=0, a_max=max_shape[0])

        preds.extend((px, py))

    return np.stack(preds, axis=-1)


ARCFACE_DISTANCE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def norm_crop(image, keypoints, image_size=112):
    M = estimate_norm(keypoints, image_size)
    return cv2.warpAffine(image, M, (image_size, image_size), borderValue=0.0)


def estimate_norm(
    keypoints: np.ndarray,
    image_size: int = 112,
    expected_size: int = 112,
    expected_keypoints_shape: tuple[int, int] = (5, 2),
) -> np.ndarray:
    if image_size % 112 != 0:
        raise RuntimeError(
            f"Expected the image size multiple of {expected_size}",
        )

    if keypoints.shape != expected_keypoints_shape:
        raise RuntimeError(
            f"Expected the keypoints of shape {expected_keypoints_shape}",
        )

    ratio = float(image_size) / expected_size
    dst = ARCFACE_DISTANCE * ratio
    similarity = SimilarityTransform()
    similarity.estimate(keypoints, dst)
    return similarity.params[0:2, :]
