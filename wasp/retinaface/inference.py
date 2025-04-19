import pathlib
from typing import Callable, TypeVar

import click
import cv2
import numpy as np
import torch
from torchvision.ops import nms

from wasp.retinaface.augmentations import test
from wasp.retinaface.data import (
    DEFAULT_MAPPING,
    Annotation,
    DetectionTargets,
    Sample,
    read_dataset,
)
from wasp.retinaface.encode import decode
from wasp.retinaface.priors import priorbox
from wasp.retinaface.ssd import RetinaNetPure
from wasp.retinaface.train import DedetectionModel
from wasp.retinaface.visualize.plot import plot, to_local


def batch_to_sample(
    boxes: np.ndarray,
    classes: np.ndarray,
    keypoints: np.ndarray,
    depths: np.ndarray,
) -> Sample:
    score, label = classes.argmax(axis=-1)
    predictions = zip(
        boxes.tolist(),
        label.reshape(-1, 1).tolist(),
        score.reshape(-1, 1).tolist(),
    )

    return Sample(
        file_name="inference",
        annotations=[
            Annotation(
                bbox=box,
                label=label,
                score=score,
                landmarks=[],
            )
            for box, label, score in predictions
        ],
    )


T = TypeVar("T")


def pred_to_labels(
    y_pred: DetectionTargets,
    anchors: torch.Tensor,
    convert: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], T],
    variances: tuple[float, float] = (0.1, 0.2),
    nms_threshold: float = 0.4,
    confidence_threshold: float = 0.1,
) -> list[T]:
    confidence = torch.nn.functional.softmax(y_pred.classes, dim=-1)
    total: list[T] = []
    for batch_id, y_pred_boxes in enumerate(y_pred.boxes):
        boxes_pred = decode(
            y_pred_boxes,
            anchors,
            variances,
        )
        # NB: it's desired to start class_ids from 0,
        # 0 is for background it's not included
        scores = confidence[batch_id][:, 1:]

        valid_index = torch.where((scores > confidence_threshold).any(-1))[0]

        # NMS doesn't accept fp16 inputs
        boxes_pred = boxes_pred[valid_index].float()
        scores = scores[valid_index].float()
        probs_pred, label_pred = scores.max(dim=-1)

        # do NMS
        keep = nms(boxes_pred, probs_pred, nms_threshold)
        boxes_pred = boxes_pred[keep, :].cpu().detach().numpy()
        probs_pred = scores[keep].cpu().detach().numpy()
        label_pred = label_pred[keep].cpu().detach().numpy()
        total.append(
            convert(
                boxes_pred,
                probs_pred,
                np.ndarray([]),
                np.ndarray([]),
            )
        )
    return total


def infer(
    image: np.ndarray,
    resolution: tuple[int, int],
    model: DedetectionModel,
) -> list[Annotation]:
    h, w = image.shape[:2]
    transform = test(resolution=resolution)
    image = transform(image=image)["image"]
    # There is only one image in the batch
    y_pred = model(image.unsqueeze(0))
    samples = pred_to_labels(y_pred, model.priors, batch_to_sample)
    annotations = samples[0].annotations

    # Convert to original image size
    for annotation in annotations:
        annotation.bbox = tuple(
            np.array(annotation.bbox) * np.array([w, h, w, h])  # type: ignore
        )

    return annotations


def true_to_labels(y_true: DetectionTargets) -> list[DetectionTargets]:
    return [
        DetectionTargets(
            boxes=y_true.boxes[batch_id],
            classes=y_true.classes[batch_id],
            keypoints=y_true.keypoints[batch_id],
            depths=y_true.depths[batch_id],
        )
        for batch_id in range(len(y_true.boxes))
    ]


def pred_to_evaluation(pred: DetectionTargets[np.ndarray]) -> np.ndarray:
    score, label = pred.classes.max(dim=-1)
    return np.concatenate(
        (
            pred.boxes.reshape(-1, 4),
            label.reshape(-1, 1),
            score.reshape(-1, 1),
        ),
        axis=1,
    )


def true_to_evaluation(true: DetectionTargets[torch.Tensor]) -> np.ndarray:
    boxes_true = true.boxes.cpu().numpy()
    label_true = true.classes.cpu().numpy()
    output = np.zeros((boxes_true.shape[0], 7), dtype=np.float32)
    output[:, :4] = boxes_true[:, :4]
    output[:, 4] = label_true[:, -1] - 1
    return output


def prepare_outputs(
    y_pred: DetectionTargets,
    y_true: DetectionTargets,
    anchors: torch.Tensor,
) -> list[tuple[np.ndarray, np.ndarray]]:
    true = true_to_labels(y_true)
    pred = pred_to_labels(
        y_pred,
        anchors,
        convert=DetectionTargets[np.ndarray],
    )
    total: list[tuple[np.ndarray, np.ndarray]] = []
    total.extend(
        (pred_to_evaluation(boxes_pred), true_to_evaluation(boxes_true))
        for boxes_pred, boxes_true in zip(pred, true)
    )
    return total


@click.command()
@click.option(
    "--dataset",
    type=click.Path(
        exists=True,
        path_type=pathlib.Path,
    ),
    default="wider/train.json",
)
def main(
    dataset,
    resolution: tuple[int, int] = (640, 640),
):
    mapping = DEFAULT_MAPPING
    model = DedetectionModel(
        RetinaNetPure(resolution, n_classes=max(mapping.values()) + 1),
        priors=priorbox(
            min_sizes=[[16, 32], [64, 128], [256, 512]],
            steps=[8, 16, 32],
            clip=False,
            image_size=resolution,
        ),
    )
    samples = read_dataset(dataset)
    for sample in samples:
        image = cv2.imread(to_local(sample.file_name))
        predicted = infer(image, resolution, model)
        true = plot(image, annotations=sample.annotations)
        print(sample.annotations)
        print(predicted)
        pred = plot(true, annotations=predicted)
        cv2.imshow("image", pred)
        cv2.waitKey()
        break


if __name__ == "__main__":
    main()
