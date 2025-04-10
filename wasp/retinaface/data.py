import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import albumentations as albu
import cv2
import numpy as np
import torch
from dacite import Config, from_dict
from dataclasses_json import dataclass_json
from environs import Env
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data

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


def to_annotations(
    sample: Sample,
    mapping: dict[str, int],
) -> LearningAnnotation:
    bboxes = []
    landmarks = []
    label_ids = []
    depths = []

    for label in sample.annotations:
        bboxes.append(label.bbox)

        lm = np.full((1, 10), np.nan)
        if label.landmarks:
            lm = np.array(label.landmarks).reshape(-1, 10)
        landmarks.append(lm[0])

        label_id = mapping.get(label.label, 0)
        label_ids.append([label_id])
        depths.append([-1, -1])

    return LearningAnnotation(
        boxes=np.array(bboxes),
        keypoints=np.array(landmarks),
        classes=np.array(label_ids),
        depths=np.array(depths),
    )


def norm(boxes, w, h):
    boxes[:, 0::2] = boxes[:, 0::2] / w
    boxes[:, 1::2] = boxes[:, 1::2] / h
    return boxes


def clip(
    bbox: np.ndarray,
    w: int,
    h: int,
) -> np.ndarray:
    x_min = np.clip(bbox[:, 0], 0, w - 1)
    y_min = np.clip(bbox[:, 1], 0, h - 1)
    x_max = np.clip(bbox[:, 2], x_min + 1, w - 1)
    y_max = np.clip(bbox[:, 3], y_min, h - 1)
    return np.stack([x_min, y_min, x_max, y_max], axis=1)


class FaceDetectionDataset(data.Dataset):
    def __init__(
        self,
        label_path: Path | str,
        transform: albu.Compose,
        mapping: Optional[dict[str, int]] = None,
    ) -> None:
        self.mapping = mapping or DEFAULT_MAPPING
        self.transform = transform
        self.labels = read_dataset(
            to_local(label_path, LOCAL_STORAGE_LOCATION),
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> LearningSample:
        sample = self.labels[index]
        annotation = to_annotations(sample, self.mapping)

        image = load_rgb(to_local(sample.file_name, LOCAL_STORAGE_LOCATION))
        h, w = image.shape[:2]

        annotation.boxes = norm(clip(annotation.boxes, w=w, h=h), w=w, h=h)
        annotation.keypoints = norm(annotation.keypoints, w=w, h=h)

        image = self.transform(
            image=image,
            category_ids=np.ones(len(annotation.boxes)),
        )["image"]

        return LearningSample(
            image=to_tensor(image),
            annotation=annotation,
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


@dataclass
class DetectionTask:
    boxes: torch.Tensor
    classes: torch.Tensor
    keypoints: torch.Tensor
    depths: torch.Tensor


@dataclass
class Batch:
    image: torch.Tensor
    annotation: torch.Tensor
    file_names: list[str]


def detection_collate(batch: List[LearningSample]) -> Batch:
    images = torch.stack([sample.image for sample in batch])
    annotations = {
        key: pad_sequence(
            [torch.tensor(asdict(sample.annotation)[key]) for sample in batch],
            batch_first=True,
            padding_value=0,
        )
        for key in asdict(batch[0].annotation).keys()
    }
    file_names = [sample.file for sample in batch]
    return Batch(images, DetectionTask(**annotations), file_names)
