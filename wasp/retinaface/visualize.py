import pathlib

import click
import cv2
import environs as Env
import matplotlib.pyplot as plt
import numpy as np

from wasp.retinaface.data import Annotation, read_dataset

env = Env()
env.read_env()


def to_local(filename):
    return filename.replace(env.str("PRIVATE_STORAGE_LOCATION"), "")


def plot(
    image: np.ndarray,
    annotations: list[Annotation],
) -> np.ndarray:
    vis_image = image.copy()
    print(annotations)

    for annotation in annotations:
        colors = [
            (255, 0, 0),
            (128, 255, 0),
            (255, 178, 102),
            (102, 128, 255),
            (0, 255, 255),
        ]

        for landmark_id, (x, y) in enumerate(annotation.landmarks):
            vis_image = cv2.circle(
                vis_image,
                (int(x), int(y)),
                radius=3,
                color=colors[landmark_id],
                thickness=3,
            )

        x_min, y_min, x_max, y_max = (int(tx) for tx in annotation.bbox)
        x_min = np.clip(x_min, 0, x_max - 1)
        y_min = np.clip(y_min, 0, y_max - 1)

        vis_image = cv2.rectangle(
            vis_image,
            (x_min, y_min),
            (x_max, y_max),
            color=(0, 255, 0),
            thickness=2,
        )

    return vis_image


@click.command()
@click.option(
    "--dataset",
    type=click.Path(
        exists=True,
        path_type=pathlib.Path,
    ),
    default="wider/test.json",
)
def main(dataset):
    labels = read_dataset(dataset)
    for sample in labels:
        image = cv2.imread(to_local(sample.file_name))
        plt.imshow(plot(image, annotations=sample.annotations))
        plt.show()


if __name__ == "__main__":
    main()
