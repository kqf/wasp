import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Generic, List, Optional, TypeVar

import albumentations as albu
import albumentations as alb
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


@dataclass
class WeightedLoss:
    loss: torch.nn.Module
    weight: float = 1.0
    enc_pred: Callable = lambda x, _: x
    enc_true: Callable = lambda x, _: x
    needs_negatives: bool = False

    def __call__(self, y_pred, y_true, anchors):
        y_pred_encoded = self.enc_pred(y_pred, anchors)
        y_true_encoded = self.enc_true(y_true, anchors)
        return self.weight * self.loss(y_pred_encoded, y_true_encoded)


T = TypeVar(
    "T",
    np.ndarray,
    torch.Tensor,
    Optional[WeightedLoss],
)


@dataclass
class DetectionTargets(Generic[T]):
    boxes: T
    classes: T
    keypoints: T
    depths: T


# A single element in the batch
@dataclass
class BatchElement(Generic[T]):
    file: str
    image: torch.Tensor
    annotation: DetectionTargets[T]


# Stacked BatchElements along batch dimension
@dataclass
class Batch(Generic[T]):
    files: list[str]
    image: torch.Tensor
    annotation: DetectionTargets[T]


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


def apply(
    transform: alb.Compose,
    image: np.ndarray,
    annotations: DetectionTargets[np.ndarray],
) -> tuple[np.ndarray, DetectionTargets[np.ndarray]]:
    # TODO: Fix for the annotation errors
    annotations.boxes = clip(annotations.boxes, image.shape[1], image.shape[0])
    transformed = transform(
        image=image,
        bboxes=annotations.boxes.reshape(-1, 4).tolist(),
        keypoints=annotations.keypoints.reshape(-1, 2).tolist(),
        category_ids=[int(cls[0]) for cls in annotations.classes],
    )

    new_annotations = DetectionTargets(
        boxes=np.array(transformed["bboxes"], dtype=np.float32),
        keypoints=np.array(transformed["keypoints"], dtype=np.float32),
        classes=annotations.classes,
        depths=annotations.depths,
    )

    h, w = transformed["image"].shape[:2]
    new_annotations.boxes = norm(new_annotations.boxes, w=w, h=h)
    new_annotations.keypoints = norm(
        new_annotations.keypoints.reshape(len(new_annotations.boxes), -1, 2),
        w=w,
        h=h,
    )

    return transformed["image"], new_annotations


def norm(boxes, w, h):
    boxes = boxes.astype(np.float32)
    boxes[..., 0::2] = boxes[..., 0::2] / w
    boxes[..., 1::2] = boxes[..., 1::2] / h
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

    def __getitem__(self, index: int) -> BatchElement[torch.Tensor]:
        sample = self.labels[index]
        annotation = to_annotations(sample, self.mapping)
        image = load_rgb(to_local(sample.file_name, LOCAL_STORAGE_LOCATION))

        image_t, annotation_t = apply(self.transform, image, annotation)

        return BatchElement(
            file=sample.file_name,
            image=to_tensor(image_t),
            annotation=annotation_t,
        )


def detection_collate(batch: List[BatchElement]) -> Batch:
    images = torch.stack([sample.image for sample in batch])
    annotations = {
        field.name: pad_sequence(
            [torch.tensor(getattr(e.annotation, field.name)) for e in batch],
            batch_first=True,
            padding_value=0,
        )
        for field in fields(batch[0].annotation)
    }
    files = [sample.file for sample in batch]
    return Batch(files, images, DetectionTargets(**annotations))
