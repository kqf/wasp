import pathlib

import click
import cv2
import matplotlib.pyplot as plt

from wasp.retinaface.data import Annotation, read_dataset, to_annotations
from wasp.retinaface.preprocess import compose, normalize, preprocess
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
        if i > 10:
            break
        image = cv2.imread(to_local(sample.file_name))
        h, w, _ = image.shape
        annotations = to_annotations(sample, h, w)
        augment = compose(normalize, preprocess)
        timage, annotations = augment(image, annotations, h)
        annotations = annotations * timage.shape[0]
        boxes = annotations[:, :4].tolist()
        keypoints = annotations[:, 4:14].reshape(-1, 5, 2).tolist()
        transformed = [Annotation(b, k) for b, k in zip(boxes, keypoints)]

        plt.imshow(plot(timage, annotations=transformed))
        plt.show()


if __name__ == "__main__":
    main()
