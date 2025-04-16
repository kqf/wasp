import albumentations as alb
from albumentations import (
    Compose,
    HueSaturationValue,
    Normalize,
    RandomBrightnessContrast,
    RandomGamma,
    Resize,
)
from albumentations.pytorch import ToTensorV2


def train(resolution: tuple[int, int]) -> Compose:
    return Compose(
        bbox_params=alb.BboxParams(
            format="pascal_voc",
            label_fields=["category_ids"],
            min_visibility=0.3,
        ),
        keypoint_params=alb.KeypointParams(
            format="xy",
            remove_invisible=False,
        ),
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
            ToTensorV2(),
        ],
    )


def valid(resolution: tuple[int, int]) -> Compose:
    return Compose(
        bbox_params=alb.BboxParams(
            format="pascal_voc",
            label_fields=["category_ids"],
            min_visibility=0.3,
        ),
        keypoint_params=alb.KeypointParams(
            format="xy",
            remove_invisible=False,
        ),
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
            ToTensorV2(),
        ],
    )


def test(resolution: tuple[int, int]) -> Compose:
    return Compose(
        bbox_params=alb.BboxParams(
            format="pascal_voc",
            label_fields=["category_ids"],
            min_visibility=0.3,
        ),
        keypoint_params=alb.KeypointParams(
            format="xy",
            remove_invisible=False,
        ),
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
            ToTensorV2(),
        ],
    )
