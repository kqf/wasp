import pathlib

import click
import cv2
import matplotlib.pyplot as plt
import numpy as np

from wasp.retinaface.augmentations import train
from wasp.retinaface.data import Annotation, read_dataset
from wasp.retinaface.visualize.plot import plot, to_local


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
        if i < 2:
            continue
        image = cv2.imread(to_local(sample.file_name))
        transform = train()
        boxes, keypoints = sample.flatten()
        print(np.asarray(keypoints), np.ones((len(boxes))))
        print(np.asarray(keypoints).reshape(-1, 2))
        sample = transform(
            image=image,
            bboxes=np.asarray(boxes),
            category_ids=np.ones(len(boxes)),
            keypoints=np.asarray(keypoints).reshape(-1, 2),
        )
        image = sample["image"]
        boxes = sample["bboxes"]
        keypoints = np.asarray(sample["keypoints"]).reshape(-1, 5, 2)
        transformed = [Annotation(b, k) for b, k in zip(boxes, keypoints)]

        plt.imshow(plot(image, annotations=transformed))
        plt.show()
        break


if __name__ == "__main__":
    main()
