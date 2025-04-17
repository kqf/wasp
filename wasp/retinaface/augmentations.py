from dataclasses import dataclass

import albumentations as alb
from albumentations import (
    Compose,
    HueSaturationValue,
    RandomBrightnessContrast,
    RandomGamma,
    Resize,
)
from albumentations.pytorch import ToTensorV2


@dataclass
class AugmentationParams:
    bbox_params: alb.BboxParams = alb.BboxParams(
        format="pascal_voc",
        label_fields=["category_ids"],
        min_visibility=1.0,
    )
    keypoint_params = alb.KeypointParams(
        format="xy",
        remove_invisible=False,
    )


default_params = AugmentationParams()


def train(resolution: tuple[int, int]) -> Compose:
    return Compose(
        bbox_params=default_params.bbox_params,
        keypoint_params=default_params.keypoint_params,
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
            ToTensorV2(),
        ],
    )


def valid(resolution: tuple[int, int]) -> Compose:
    return Compose(
        bbox_params=default_params.bbox_params,
        keypoint_params=default_params.keypoint_params,
        p=1,
        transforms=[
            Resize(*resolution),
            ToTensorV2(),
        ],
    )


def test(resolution: tuple[int, int]) -> Compose:
    return Compose(
        # bbox_params=default_params.bbox_params,
        # keypoint_params=default_params.keypoint_params,
        p=1,
        transforms=[
            Resize(*resolution),
            ToTensorV2(),
        ],
    )
