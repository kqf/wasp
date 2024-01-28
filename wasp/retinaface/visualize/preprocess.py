import pathlib

import click
import cv2
import matplotlib.pyplot as plt
import numpy as np

from wasp.retinaface.data import Annotation, Sample, read_dataset, trimm_boxes
from wasp.retinaface.preprocess import preprocess
from wasp.retinaface.visualize.plot import plot, to_local


def with_masks(keypoints):
    mask = keypoints < 0
    return mask, keypoints.clip(0, 1024)


def to_annotations(sample: Sample, image_width, image_height) -> np.ndarray:
    num_annotations = 4 + 10 + 1
    annotations = np.zeros((0, num_annotations))

    for label in sample.annotations:
        annotation = np.zeros((1, num_annotations))

        annotation[0, :4] = trimm_boxes(
            label.bbox,
            image_width=image_width,
            image_height=image_height,
        )

        if label.landmarks:
            landmarks = np.array(label.landmarks)
            # landmarks
            annotation[0, 4:14] = landmarks.reshape(-1, 10)
            annotation[0, 14] = -1 if annotation[0, 4] < 0 else 1
        annotations = np.append(annotations, annotation, axis=0)
    return annotations


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
        w, h, _ = image.shape
        annotations = to_annotations(sample, w, h)
        timage, annotations = preprocess(image, annotations, h)
        annotations = annotations * h
        boxes = annotations[:, :4].tolist()
        keypoints = annotations[:, 4:14].reshape(-1, 5, 2).tolist()
        print(boxes)
        transformed = [Annotation(b, k) for b, k in zip(boxes, keypoints)]

        plt.imshow(plot(timage, annotations=transformed))
        plt.show()
        break


if __name__ == "__main__":
    main()
