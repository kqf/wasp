import random

import numpy as np


def random_crop(*args, **kwargs):
    return args


def _pad_to_square(image: np.ndarray, pad_image_flag: bool) -> np.ndarray:
    if not pad_image_flag:
        return image
    height, width = image.shape[:2]
    long_side = max(width, height)
    image_t = np.zeros((long_side, long_side, 3), dtype=image.dtype)
    image_t[:height, :width] = image
    return image_t


def random_horizontal_flip(
    image: np.ndarray,
    boxes: np.ndarray,
    landms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = image.shape[1]
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]

        # landm
        landms = landms.copy()
        landms = landms.reshape([-1, 5, 2])
        landms[:, :, 0] = width - landms[:, :, 0]
        flip_landmark(landms, 1, 0)
        flip_landmark(landms, 4, 3)
        landms = landms.reshape([-1, 10])

    return image, boxes, landms


# TODO Rename this here and in `random_horizontal_flip`
def flip_landmark(landms, arg1, arg2):
    tmp = landms[:, arg1, :].copy()
    landms[:, arg1, :] = landms[:, arg2, :]
    landms[:, arg2, :] = tmp


class Preproc:
    def __init__(self, img_dim: int) -> None:
        self.img_dim = img_dim

    def __call__(
        self,
        image: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if targets.shape[0] == 0:
            raise ValueError("this image does not have gt")

        boxes = targets[:, :4].copy()
        landmarks = targets[:, 4:-1].copy()
        labels = targets[:, -1:].copy()

        image_t, boxes_t, labels_t, landmarks_t, pad_image_flag = random_crop(
            image, boxes, labels, landmarks, self.img_dim
        )

        image_t = _pad_to_square(image_t, pad_image_flag)
        image_t, boxes_t, landmarks_t = random_horizontal_flip(
            image_t, boxes_t, landmarks_t
        )
        height, width = image_t.shape[:2]

        boxes_t[:, 0::2] = boxes_t[:, 0::2] / width
        boxes_t[:, 1::2] = boxes_t[:, 1::2] / height

        landmarks_t[:, 0::2] = landmarks_t[:, 0::2] / width
        landmarks_t[:, 1::2] = landmarks_t[:, 1::2] / height

        targets_t = np.hstack((boxes_t, landmarks_t, labels_t))

        return image_t, targets_t
