import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Generic, List, Optional, TypeVar

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


CLOUD_STORAGE_LOCATION = env.str("PRIVATE_STORAGE_LOCATION")
LOCAL_STORAGE_LOCATION = env.str("LOCAL_STORAGE_LOCATION")


def to_local(filename: Path | str, local: str = "") -> str:
    return str(filename).replace(CLOUD_STORAGE_LOCATION, local)


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


T = TypeVar("T", np.ndarray, torch.Tensor)


@dataclass
class DetectionTargets(Generic[T]):
    boxes: T
    classes: T
    keypoints: T
    depths: T


@dataclass
class Batch(Generic[T]):
    image: torch.Tensor
    annotation: DetectionTargets[T]
    files: list[str]


def to_annotations(
    sample: Sample,
    mapping: dict[str, int],
) -> DetectionTargets[np.ndarray]:
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

    return DetectionTargets(
        boxes=np.array(bboxes),
        keypoints=np.array(landmarks),
        classes=np.array(label_ids),
        depths=np.array(depths),
    )


def norm(boxes, w, h):
    boxes = boxes.astype(np.float32)
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

    def __getitem__(self, index: int) -> Batch[torch.Tensor]:
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

        return Batch(
            image=to_tensor(image),
            annotation=annotation,
            files=[sample.file_name],
        )


def detection_collate(batch: List[Batch[torch.tensor]]) -> Batch[torch.Tensor]:
    images = torch.stack([sample.image for sample in batch])
    annotations = {
        key: pad_sequence(
            [torch.tensor(asdict(sample.annotation)[key]) for sample in batch],
            batch_first=True,
            padding_value=0,
        )
        for key in asdict(batch[0].annotation).keys()
    }
    files = [sample.files[0] for sample in batch]
    return Batch(images, DetectionTargets(**annotations), files)
