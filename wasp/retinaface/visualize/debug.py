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
    load_rgb,
    read_dataset,
    to_annotations,
)
from wasp.retinaface.visualize.plot import plot, to_local


def to_labels(
    detection_targets: DetectionTargets[np.ndarray],
    mapping: dict[int, str],
) -> list[Annotation]:
    annotations = []
    for i in range(len(detection_targets.boxes)):
        bbox = tuple(detection_targets.boxes[i].tolist())
        keypoints = detection_targets.keypoints[i].reshape(-1, 2).tolist()
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
            alb.HorizontalFlip(p=0.5),
            alb.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.7,
            ),
            alb.RandomBrightnessContrast(p=0.2),
        ],
        bbox_params=alb.BboxParams(
            format="pascal_voc",  # absolute pixel format
            label_fields=["category_ids"],
            min_visibility=0.3,
        ),
        keypoint_params=alb.KeypointParams(
            format="xy",  # For individual (x, y) keypoints
            remove_invisible=False,
        ),
    )


def apply(
    transform: alb.Compose,
    image: np.ndarray,
    annotations: DetectionTargets[np.ndarray],
) -> tuple[np.ndarray, DetectionTargets[np.ndarray]]:
    transformed = transform(
        image=image,
        bboxes=annotations.boxes.reshape(-1, 4).tolist(),
        keypoints=annotations.keypoints.reshape(-1, 2).tolist(),
        category_ids=[int(cls[0]) for cls in annotations.classes],
    )
    new_annotations = DetectionTargets(
        boxes=np.array(transformed["bboxes"], dtype=np.float32),
        keypoints=np.array(transformed["keypoints"], dtype=np.float32).reshape(
            len(transformed["bboxes"]), -1, 2
        ),
        classes=annotations.classes,
        depths=annotations.depths,
    )

    return transformed["image"], new_annotations


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
        processed_annotations = to_labels(annotations, mapping=DEFAULT_MAPPING)
        original = plot(image, processed_annotations)
        cv2.imshow("image", original)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
