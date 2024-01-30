import cv2
import numpy as np
from environs import Env

from wasp.retinaface.data import Annotation

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
            (0, 0, 255),  # left eye
            (0, 255, 0),  # right eye
            (255, 0, 0),  # nose
            (0, 128, 255),  # left mouth
            (255, 255, 128),  # right mouth
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
