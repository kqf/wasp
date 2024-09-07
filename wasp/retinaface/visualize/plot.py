import os

import cv2
import numpy as np
import torch
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


def denormalize(image):
    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        dtype=image.dtype,
        device=image.device,
    )
    std = torch.tensor(
        [0.229, 0.224, 0.225],
        dtype=image.dtype,
        device=image.device,
    )

    # Reverse normalization: x' = (x * std) + mean
    image = (image * std[:, None, None]) + mean[:, None, None]
    image = image.clamp(0, 1)  # Clamping to [0, 1] range
    return image * 255.0


def to_image(image):
    return denormalize(image).permute(1, 2, 0).cpu().numpy().astype(np.uint8)


def draw_anchors_on_image(image, anchors, y_true_, y_pred_, odir):
    image = to_image(image)
    anchors = anchors.cpu().numpy()

    # Convert image from torch (C, H, W) to OpenCV (H, W, C) and BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    height, width, _ = image.shape

    def rectangle(image, box, color):
        pt1 = (int(box[0] * width), int(box[1] * height))
        pt2 = (int(box[2] * width), int(box[3] * height))
        return cv2.rectangle(image, pt1, pt2, color, 2)

    # Draw each anchor box on the image
    for i, box in enumerate(anchors):
        image = rectangle(image, box, color=(255, 255, 255))

    for i, box in enumerate(y_true_):
        image = rectangle(image, box, color=(0, 0, 0))

    for i, box in enumerate(y_pred_):
        image = rectangle(image, box, color=(255, 0, 0))

    # Save the image
    cv2.imwrite(odir, image)


def draw_anchors(images, anchors, odir="anchors"):
    os.makedirs(odir, exist_ok=True)
    for i, (image, anchors) in enumerate(zip(images, anchors)):
        draw_anchors_on_image(image, anchors, f"{odir}/image-{i}-anchors.jpg")
