import pathlib

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


def pred_to_labels(
    y_pred: DetectionTargets,
    anchors: torch.Tensor,
    variances: tuple[float, float] = (0.1, 0.2),
    nms_threshold: float = 0.4,
    confidence_threshold: float = 0.1,
) -> list[Sample]:
    confidence = torch.nn.functional.softmax(y_pred.classes, dim=-1)
    total: list[Sample] = []
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
        probs_pred = probs_pred[keep].cpu().detach().numpy()
        label_pred = label_pred[keep].cpu().detach().numpy()
        predictions = zip(
            boxes_pred.tolist(),
            label_pred.reshape(-1, 1).tolist(),
            probs_pred.reshape(-1, 1).tolist(),
        )
        total.extend(
            Sample(
                file_name="inference",
                annotations=[
                    Annotation(
                        bbox=box,
                        label=label,
                        score=score,
                        landmarks=[],
                    )
                ],
            )
            for box, label, score in predictions
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
    samples = pred_to_labels(y_pred, model.priors)
    annotations = samples[0].annotations

    # Convert to original image size
    for annotation in annotations:
        annotation.bbox = tuple(
            np.array(annotation.bbox) * np.array([w, h, w, h])  # type: ignore
        )

    return annotations


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
