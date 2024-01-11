from albumentations import (
    BboxParams,
    Compose,
    HueSaturationValue,
    Normalize,
    RandomBrightnessContrast,
    RandomGamma,
    Resize,
)


def train() -> Compose:
    return Compose(
        # bbox_params=None,
        bbox_params=BboxParams(
            format="pascal_voc",
            label_fields=["category_ids"],
        ),
        # keypoint_params=None,
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
            # Resize(height=256, width=256, p=1),
            Normalize(
                always_apply=False,
                max_pixel_value=255.0,
                mean=[0.485, 0.456, 0.406],
                p=1,
                std=[0.229, 0.224, 0.225],
            ),
        ],
    )


def valid() -> Compose:
    return Compose(
        bbox_params=None,
        keypoint_params=None,
        p=1,
        transforms=[
            Resize(height=256, width=256, p=1),
            Normalize(
                always_apply=False,
                max_pixel_value=255.0,
                mean=[0.485, 0.456, 0.406],
                p=1,
                std=[0.229, 0.224, 0.225],
            ),
        ],
    )


def test() -> Compose:
    return Compose(
        bbox_params=None,
        keypoint_params=None,
        p=1,
        transforms=[
            Normalize(
                always_apply=False,
                max_pixel_value=255.0,
                mean=[0.485, 0.456, 0.406],
                p=1,
                std=[0.229, 0.224, 0.225],
            )
        ],
    )
