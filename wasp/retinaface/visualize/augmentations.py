import pathlib

import click
import cv2
import matplotlib.pyplot as plt

from wasp.retinaface.data import read_dataset
from wasp.retinaface.visualize import plot, to_local


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
