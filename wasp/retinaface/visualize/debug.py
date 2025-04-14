import pathlib

import albumentations as alb
import click
import cv2
import numpy as np

from wasp.retinaface.data import (
    DEFAULT_MAPPING,
    Annotation,
    DetectionTargets,
    Sample,
    apply,
    load_rgb,
    read_dataset,
    to_annotations,
)
from wasp.retinaface.visualize.plot import plot, to_local


def un_norm(boxes, w, h):
    boxes = boxes.astype(np.float32)
    boxes[..., 0::2] = boxes[..., 0::2] * w
    boxes[..., 1::2] = boxes[..., 1::2] * h
    return boxes


def to_labels(
    image: np.ndarray,
    detection_targets: DetectionTargets[np.ndarray],
    mapping: dict[int, str],
) -> list[Annotation]:
    annotations = []
    h, w = image.shape[:2]
    boxes = un_norm(detection_targets.boxes, w, h)
    keypoints = un_norm(detection_targets.keypoints, w, h)

    for i in range(len(boxes)):
        bbox = tuple(boxes[i].tolist())
        keypoints = keypoints[i].reshape(-1, 2).tolist()
        label_id = int(detection_targets.classes[i][0])
        label = mapping.get(label_id, "unknown")
        annotations.append(
            Annotation(
                bbox=bbox,  # type: ignore
                landmarks=keypoints,
                label=label,
            )
        )
    return annotations


def build_geometric_augs() -> alb.Compose:
    return alb.Compose(
        [
            # Horizontal flip
            alb.HorizontalFlip(p=0.5),
            # Random crop that keeps at least part of the bounding boxes safe
            alb.CropAndPad(
                px=(-50, 50),  # crop or pad up to 50px on each side
                sample_independently=True,
                pad_mode=cv2.BORDER_REFLECT_101,
                pad_cval=0,
                p=0.5,
            ),
            # Zoom in/out (scaling without changing image size)
            alb.Affine(
                scale=(0.8, 1.2),  # zoom out to zoom in
                fit_output=False,  # keep original canvas size
                mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
            # Random rotation in-place (no image scaling)
            alb.Affine(
                rotate=(-15, 15),
                scale=1.0,  # no scale
                fit_output=False,
                mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
            # Slight color/contrast adjustment
            alb.RandomBrightnessContrast(p=0.2),
        ],
        bbox_params=alb.BboxParams(
            format="pascal_voc",
            label_fields=["category_ids"],
            min_visibility=0.3,
        ),
        keypoint_params=alb.KeypointParams(
            format="xy",
            remove_invisible=False,
        ),
    )


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
    index = 0
    sample: Sample = labels[index]

    for _ in range(10):
        file_name = to_local(sample.file_name)
        image = load_rgb(file_name)

        # image_h, image_w = image.shape[:2]
        annotations = to_annotations(sample, mapping=DEFAULT_MAPPING)
        image, annotations = apply(build_geometric_augs(), image, annotations)
        processed_annotations = to_labels(
            image,
            annotations,
            mapping=DEFAULT_MAPPING,
        )
        original = plot(image, processed_annotations)
        cv2.imshow("image", original)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
