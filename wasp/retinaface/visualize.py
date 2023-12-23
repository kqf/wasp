import json
from typing import Any, Dict, List

import click
import cv2
import matplotlib.pyplot as plt
import numpy as np


def to_local(filename):
    return filename.replace("s3://leiaml-datasets/face-tracking/v0.0.1/", "")


def vis_annotations(
    image: np.ndarray,
    annotations: List[Dict[str, Any]],
) -> np.ndarray:
    vis_image = image.copy()
    print(annotations)

    for i, annotation in enumerate(annotations):
        landmarks = annotation["landmarks"]

        colors = [
            (255, 0, 0),
            (128, 255, 0),
            (255, 178, 102),
            (102, 128, 255),
            (0, 255, 255),
        ]

        for landmark_id, (x, y) in enumerate(landmarks):
            vis_image = cv2.circle(
                vis_image,
                (int(x), int(y)),
                radius=3,
                color=colors[landmark_id],
                thickness=3,
            )

        x_min, y_min, x_max, y_max = (int(tx) for tx in annotation["bbox"])

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
    "--labels",
    type=click.Path(exists=True),
    default="wider/train.json",
)
def main(labels):
    with open(labels) as f:
        annotations = json.load(f)
    for entry in annotations:
        image = cv2.imread(to_local(entry["file_name"]))
        plt.imshow(vis_annotations(image, annotations=entry["annotations"]))
        plt.show()


if __name__ == "__main__":
    main()
