import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import albumentations as albu
import cv2
import numpy as np
import torch
from dacite import Config, from_dict
from dataclasses_json import dataclass_json
from environs import Env
from torch.utils import data

from wasp.retinaface.preprocess import preprocess

env = Env()
env.read_env()


def to_local(filename: Path | str, local: str = "") -> str:
    return str(filename).replace(env.str("PRIVATE_STORAGE_LOCATION"), local)


LOCAL_STORAGE_LOCATION = env.str("LOCAL_STORAGE_LOCATION")


def to_tensor(image: np.ndarray) -> torch.Tensor:
    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
    return torch.from_numpy(image)


def load_rgb(image_path: Path | str) -> np.array:
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


AbsoluteXYXY = tuple[int, int, int, int]


@dataclass_json
@dataclass
class Annotation:
    bbox: AbsoluteXYXY
    landmarks: list
    label: str = "person"


@dataclass_json
@dataclass
class Sample:
    file_name: str
    annotations: list[Annotation]

    def flatten(self) -> tuple:
        return tuple(zip(*[(a.bbox, a.landmarks) for a in self.annotations]))


def to_sample(entry: dict[str, Any]) -> Sample:
    return from_dict(
        data_class=Sample,
        data=entry,
        config=Config(cast=[tuple]),
    )


def read_dataset(path: Path | str) -> list[Sample]:
    with open(path) as f:
        df = json.load(f)
    return [to_sample(x) for x in df]


def trimm_boxes(
    bbox: AbsoluteXYXY,
    image_width: int,
    image_height: int,
) -> AbsoluteXYXY:
    x_min, y_min, x_max, y_max = bbox

    x_min = np.clip(x_min, 0, image_width - 1)
    y_min = np.clip(y_min, 0, image_height - 1)
    x_max = np.clip(x_max, x_min + 1, image_width - 1)
    y_max = np.clip(y_max, y_min, image_height - 1)

    return x_min, y_min, x_max, y_max


def to_annotations(
    sample: Sample,
    image_w: int,
    image_h: int,
    mapping: dict[str, int],
) -> np.ndarray:
    num_annotations = 4 + 10 + 1
    annotations = np.zeros((0, num_annotations))

    for label in sample.annotations:
        annotation = np.empty((1, num_annotations))

        annotation[0, :4] = trimm_boxes(
            label.bbox,
            image_width=image_w,
            image_height=image_h,
        )

        if label.landmarks:
            landmarks = np.array(label.landmarks)
            # landmarks
            annotation[0, 4:14] = landmarks.reshape(-1, 10)
        else:
            annotation[0, 4:14] = np.nan

        # Important use either nan or 0
        annotation[0, 14] = mapping.get(label.label, 0)
        annotations = np.append(annotations, annotation, axis=0)

    return annotations


def to_dicts(annotations: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "boxes": annotations[:, :4],
        "keypoints": annotations[:, 4:14],
        "classes": annotations[:, [14]].astype(np.int64),
        "depths": annotations[:, 15:],
    }


DEFAULT_MAPPING = {
    "person": 1,
}


@dataclass
class LearningAnnotation:
    boxes: np.ndarray
    classes: np.ndarray
    keypoints: np.ndarray
    depths: np.ndarray


@dataclass
class LearningSample:
    image: np.ndarray
    file: str
    annotation: LearningAnnotation


class FaceDetectionDataset(data.Dataset):
    def __init__(
        self,
        label_path: Path | str,
        transform: albu.Compose,
        mapping: Optional[dict[str, int]] = None,
        preproc: Callable = preprocess,
        rotate90: bool = False,
    ) -> None:
        self.mapping = mapping or DEFAULT_MAPPING
        self.preproc = preproc
        self.transform = transform
        self.rotate90 = rotate90
        self.labels = read_dataset(
            to_local(label_path, LOCAL_STORAGE_LOCATION),
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> LearningSample:
        sample = self.labels[index]
        image = load_rgb(to_local(sample.file_name, LOCAL_STORAGE_LOCATION))

        image_height, image_width = image.shape[:2]
        annotations = to_annotations(
            sample,
            image_width,
            image_height,
            self.mapping,
        )

        if self.rotate90:
            image, annotations = random_rotate_90(
                image,
                annotations.astype(int),
            )

        image, annotations = self.preproc(image, annotations)

        image = self.transform(
            image=image,
            category_ids=np.ones(len(annotations)),
        )["image"]

        return LearningSample(
            image=to_tensor(image),
            annotation=LearningAnnotation(
                **to_dicts(annotations.astype(np.float32))
            ),
            file=sample.file_name,
        )


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


def stack(tensors, pad_value=0) -> torch.Tensor:
    max_length = max(tensor.shape[0] for tensor in tensors)
    return torch.stack(
        [
            torch.nn.functional.pad(
                t, (0, 0, 0, max_length - t.shape[0]), value=pad_value
            )
            for t in tensors
        ]
    )


@dataclass
class DetectionTask:
    boxes: torch.Tensor
    classes: torch.Tensor


@dataclass
class Batch:
    image: torch.Tensor
    annotation: torch.Tensor
    file_names: list[str]


def detection_collate(batch: List[LearningSample]) -> Batch:
    images = torch.stack([sample.image for sample in batch])
    annotations = {
        key: stack(
            [torch.tensor(asdict(sample.annotation)[key]) for sample in batch]
        )
        for key in asdict(batch[0].annotation).keys()
    }
    file_names = [sample.file for sample in batch]
    return Batch(images, DetectionTask(**annotations), file_names)
