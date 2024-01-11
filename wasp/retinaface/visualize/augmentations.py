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
    for sample in labels:
        image = cv2.imread(to_local(sample.file_name))
        transform = train()
        boxes, keypoints = sample.flatten()
        sample = transform(
            image=image,
            # bbox=np.asarray(boxes),
            # keypoints=np.asarray(keypoints),
        )
        image = sample["image"]
        # boxes = sample["bbox"]
        # keypoints = sample["keypoints"]
        transformed = [Annotation(b, k) for b, k in zip(boxes, keypoints)]

        plt.imshow(plot(image, annotations=transformed))
        plt.show()


if __name__ == "__main__":
    main()
