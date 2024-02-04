import pathlib

import albumentations as alb
import click
import cv2
import matplotlib.pyplot as plt
import numpy as np

from wasp.retinaface.data import Annotation, read_dataset
from wasp.retinaface.visualize.plot import plot, to_local


def train(height, width) -> alb.Compose:
    return alb.Compose(
        bbox_params=alb.BboxParams(
            format="pascal_voc",
            label_fields=["category_ids"],
        ),
        keypoint_params=alb.KeypointParams(
            format="xy",
        ),
        p=1,
        transforms=[
            # alb.RandomRotate90(p=1.0),
            # alb.RandomCrop(height, width, p=1.0),
            alb.VerticalFlip(p=1.0),
            # alb.RandomBrightnessContrast(
            #     always_apply=False,
            #     brightness_limit=0.2,
            #     contrast_limit=[0.5, 1.5],
            #     p=0.5,
            # ),
            # alb.HueSaturationValue(
            #     hue_shift_limit=20,
            #     val_shift_limit=20,
            #     p=0.5,
            # ),
            # alb.RandomGamma(gamma_limit=[80, 120], p=0.5),
            # alb.Resize(height=256, width=256, p=1),
            # alb.Normalize(
            #     always_apply=False,
            #     max_pixel_value=255.0,
            #     mean=[0.485, 0.456, 0.406],
            #     p=1,
            #     std=[0.229, 0.224, 0.225],
            # ),
        ],
    )


def with_masks(keypoints):
    mask = keypoints < 0
    return mask, keypoints.clip(0, 1024)


@click.command()
@click.option(
    "--dataset",
    type=click.Path(
        exists=True,
        path_type=pathlib.Path,
    ),
    default="wider/train.json",
)
def main(dataset):
    labels = read_dataset(dataset)
    for i, sample in enumerate(labels):
        print(i)
        image = cv2.imread(to_local(sample.file_name))
        h, w = image.shape[:2]
        transform = train(h, w)
        boxes, keypoints = sample.flatten()
        print(np.asarray(keypoints), np.ones((len(boxes))))
        print(np.asarray(keypoints).reshape(-1, 2))
        print()
        masks, clipped = with_masks(np.asarray(keypoints).reshape(-1, 2))
        sample = transform(
            image=image,
            bboxes=np.asarray(boxes),
            category_ids=np.ones(len(boxes)),
            keypoints=clipped,
        )
        image = sample["image"]
        boxes = sample["bboxes"]
        transofrmed_keypoints = np.asarray(sample["keypoints"])
        transofrmed_keypoints[masks] = -1
        keypoints = transofrmed_keypoints.reshape(-1, 5, 2)
        transformed = [Annotation(b, k) for b, k in zip(boxes, keypoints)]

        plt.imshow(plot(image, annotations=transformed))
        plt.show()
        if i > 10:
            break


if __name__ == "__main__":
    main()
