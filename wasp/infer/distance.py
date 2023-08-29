import numpy as np


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
    preds = []  # type: ignore
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]

        if max_shape is not None:
            px = np.clip(px, a_min=0, a_max=max_shape[1])
            py = np.clip(py, a_min=0, a_max=max_shape[0])

        preds.extend((px, py))

    return np.stack(preds, axis=-1)
