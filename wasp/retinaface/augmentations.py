from typing import Tuple

import numpy as np
from albumentations import (
    BboxParams,
    Compose,
    HueSaturationValue,
    KeypointParams,
    Normalize,
    RandomBrightnessContrast,
    RandomGamma,
    RandomRotate90,
    Resize,
)


def train(resolution: tuple[int, int]) -> Compose:
    return Compose(
        bbox_params=None,
        keypoint_params=None,
        p=1,
        transforms=[
            RandomBrightnessContrast(
                always_apply=False,
                brightness_limit=0.2,
                contrast_limit=[0.5, 1.5],
                p=0.5,
            ),
            HueSaturationValue(hue_shift_limit=20, val_shift_limit=20, p=0.5),
            RandomGamma(gamma_limit=[80, 120], p=0.5),
            Resize(*resolution),
            Normalize(
                always_apply=False,
                max_pixel_value=255.0,
                mean=[0.485, 0.456, 0.406],
                p=1,
                std=[0.229, 0.224, 0.225],
            ),
        ],
    )


def valid(resolution: tuple[int, int]) -> Compose:
    return Compose(
        bbox_params=None,
        keypoint_params=None,
        p=1,
        transforms=[
            Resize(*resolution),
            Normalize(
                always_apply=False,
                max_pixel_value=255.0,
                mean=[0.485, 0.456, 0.406],
                p=1,
                std=[0.229, 0.224, 0.225],
            ),
        ],
    )


def test(resolution: tuple[int, int]) -> Compose:
    return Compose(
        bbox_params=None,
        keypoint_params=None,
        p=1,
        transforms=[
            Resize(*resolution),
            Normalize(
                always_apply=False,
                max_pixel_value=255.0,
                mean=[0.485, 0.456, 0.406],
                p=1,
                std=[0.229, 0.224, 0.225],
            ),
        ],
    )


def random_rotate_90(
    image: np.ndarray,
    annotations: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    image_height, image_width = image.shape[:2]

    boxes = annotations[:, :4]
    keypoints = annotations[:, 4:-1].reshape(-1, 2)
    labels = annotations[:, -1:]

    invalid_index = keypoints.sum(axis=1) == -2

    keypoints[:, 0] = np.clip(keypoints[:, 0], 0, image_width - 1)
    keypoints[:, 1] = np.clip(keypoints[:, 1], 0, image_height - 1)

    keypoints[invalid_index] = 0

    category_ids = list(range(boxes.shape[0]))

    transform = Compose(
        [RandomRotate90(p=1)],
        keypoint_params=KeypointParams(format="xy"),
        bbox_params=BboxParams(
            format="pascal_voc",
            label_fields=["category_ids"],
        ),
    )
    transformed = transform(
        image=image,
        keypoints=keypoints.tolist(),
        bboxes=boxes.tolist(),
        category_ids=category_ids,
    )

    keypoints = np.array(transformed["keypoints"])
    keypoints[invalid_index] = -1

    keypoints = keypoints.reshape(-1, 10)
    boxes = np.array(transformed["bboxes"])
    image = transformed["image"]

    annotations = np.hstack([boxes, keypoints, labels])
    return image, annotations
