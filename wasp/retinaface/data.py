import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import albumentations as albu
import cv2
import numpy as np
import torch
from dacite import from_dict
from torch.utils import data

from wasp.retinaface.preprocess import Preproc


def to_tensor(image: np.ndarray) -> torch.Tensor:
    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
    return torch.from_numpy(image)


def load_rgb(image_path: Path | str) -> np.array:
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


@dataclass
class Annotation:
    bbox: list[int]
    landmarks: list


@dataclass
class Sample:
    file_name: str
    annotations: list[Annotation]


def read_dataset(label_path: Path) -> list[Sample]:
    with label_path.open() as f:
        df = json.load(f)
    return [from_dict(data_class=Sample, data=x) for x in df]


class FaceDetectionDataset(data.Dataset):
    def __init__(
        self,
        label_path: Path,
        image_path: Path,
        transform: albu.Compose,
        preproc: Preproc,
        rotate90: bool = False,
    ) -> None:
        self.preproc = preproc
        self.image_path = Path(image_path)
        self.transform = transform
        self.rotate90 = rotate90
        self.labels = read_dataset(label_path)

        def exists(x):
            return (image_path / x["file_name"]).exists()

        # self.labels = [x for x in labels if exists(x)]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.labels[index]
        image = load_rgb(self.image_path / sample.file_name)

        image_height, image_width = image.shape[:2]

        # annotations will have the format
        # 4: box, 10 landmarks, 1: landmarks / no landmarks
        num_annotations = 4 + 10 + 1
        annotations = np.zeros((0, num_annotations))

        for label in sample.annotations:
            annotation = np.zeros((1, num_annotations))

            x_min, y_min, x_max, y_max = label.bbox

            x_min = np.clip(x_min, 0, image_width - 1)
            y_min = np.clip(y_min, 0, image_height - 1)
            x_max = np.clip(x_max, x_min + 1, image_width - 1)
            y_max = np.clip(y_max, y_min, image_height - 1)

            annotation[0, :4] = x_min, y_min, x_max, y_max

            if label.landmarks:
                landmarks = np.array(label.landmarks)
                # landmarks
                annotation[0, 4:14] = landmarks.reshape(-1, 10)
                annotation[0, 14] = -1 if annotation[0, 4] < 0 else 1
            annotations = np.append(annotations, annotation, axis=0)

        if self.rotate90:
            image, annotations = random_rotate_90(
                image,
                annotations.astype(int),
            )

        image, annotations = self.preproc(image, annotations)

        image = self.transform(image=image)["image"]

        return {
            "image": to_tensor(image),
            "annotation": annotations.astype(np.float32),
            "file_name": sample.file_name,
        }


def random_rotate_90(
    image: np.ndarray, annotations: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    image_height, image_width = image.shape[:2]

    boxes = annotations[:, :4]
    keypoints = annotations[:, 4:-1].reshape(-1, 2)
    labels = annotations[:, -1:]

    invalid_index = keypoints.sum(axis=1) == -2

    keypoints[:, 0] = np.clip(keypoints[:, 0], 0, image_width - 1)
    keypoints[:, 1] = np.clip(keypoints[:, 1], 0, image_height - 1)

    keypoints[invalid_index] = 0

    category_ids = list(range(boxes.shape[0]))

    transform = albu.Compose(
        [albu.RandomRotate90(p=1)],
        keypoint_params=albu.KeypointParams(format="xy"),
        bbox_params=albu.BboxParams(
            format="pascal_voc",
            label_fields=["category_ids"],
        ),
    )
    transformed = transform(
        image=image,
        keypoints=keypoints.tolist(),
        bboxes=boxes.tolist(),
        category_ids=category_ids,
    )

    keypoints = np.array(transformed["keypoints"])
    keypoints[invalid_index] = -1

    keypoints = keypoints.reshape(-1, 10)
    boxes = np.array(transformed["bboxes"])
    image = transformed["image"]

    annotations = np.hstack([boxes, keypoints, labels])

    return image, annotations


def detection_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate fn for dealing with batches of images
    that have a different number of boxes.

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given
            image are stacked on 0 dim
    """
    annotation = []
    images = []
    file_names = []

    for sample in batch:
        images.append(sample["image"])
        annotation.append(torch.from_numpy(sample["annotation"]).float())
        file_names.append(sample["file_name"])

    return {
        "image": torch.stack(images),
        "annotation": annotation,
        "file_name": file_names,
    }
