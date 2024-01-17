import pathlib

import albumentations as alb
import click
import cv2
import matplotlib.pyplot as plt
import numpy as np

from wasp.retinaface.data import Annotation, read_dataset
from wasp.retinaface.visualize.plot import plot


def train() -> alb.Compose:
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
            alb.Resize(height=1024, width=1024, p=1),
        ],
    )


def with_masks(keypoints):
    mask = keypoints < 0
    return mask, keypoints.clip(0, 1024)


def main(dataset):
    labels = read_dataset(dataset)
    for i, sample in enumerate(labels):
        if i != 100:
            continue
        image = cv2.cvtColor(
            cv2.imread("couple.jpg"),
            cv2.COLOR_BGR2RGB,
        )

        transform = train()
        boxes, keypoints = sample.flatten()
        print(np.asarray(keypoints), np.ones((len(boxes))))
        print(np.asarray(keypoints).reshape(-1, 2))
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


if __name__ == "__main__":
    main()
