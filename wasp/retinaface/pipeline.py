import os
import pydoc
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from addict import Dict as Adict
from albumentations.core.serialization import from_dict
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.ops import nms

from wasp.retinaface.data import FaceDetectionDataset, detection_collate
from wasp.retinaface.matching import decode


def dpath(envv):
    def f():
        Path(os.environ[envv])

    return f


@dataclass
class Paths:
    train: Path = field(default_factory=dpath("TRAIN_IMAGE_PATH"))
    valid: Path = field(default_factory=dpath("VAL_IMAGE_PATH"))
    train_label: Path = field(default_factory=dpath("TRAIN_LABEL_PATH"))
    valid_label: Path = field(default_factory=dpath("VAL_LABEL_PATH"))


def object_from_dict(d, parent=None, **default_kwargs):
    kwargs = d.copy()
    object_type = kwargs.pop("type")
    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    if parent is not None:
        return getattr(parent, object_type)(**kwargs)  # skipcq PTC-W0034

    return pydoc.locate(object_type)(**kwargs)


class RetinaFacePipeline(pl.LightningModule):  # pylint: disable=R0901
    def __init__(
        self,
        config: Adict[str, Any],
        paths: Paths,
        model: torch.nn.Module,
        preprocessing,
        priorbox,
        build_optimizer,
        build_scheduler,
        loss,
        loss_weights,
    ) -> None:
        super().__init__()
        self.config = config
        self.paths = paths
        self.model = model
        self.prior_box = priorbox
        self.loss_weights = loss_weights
        self.loss = loss
        self.preprocessing = preprocessing
        self.build_optimizer = build_optimizer
        self.build_scheduler = build_scheduler

    def setup(self, stage=0) -> None:  # type: ignore
        self.preproc = self.preprocessing()

    def forward(
        self, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore
        return self.model(batch)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            FaceDetectionDataset(
                label_path=self.paths.train_label,
                image_path=self.paths.train,
                transform=from_dict(self.config.train_aug),
                preproc=self.preproc,
                rotate90=self.config.train_parameters.rotate90,
            ),
            batch_size=self.config.train_parameters.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            collate_fn=detection_collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            FaceDetectionDataset(
                label_path=self.paths.valid_label,
                image_path=self.paths.valid,
                transform=from_dict(self.config.val_aug),
                preproc=self.preproc,
                rotate90=self.config.val_parameters.rotate90,
            ),
            batch_size=self.config.val_parameters.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            collate_fn=detection_collate,
        )

    def configure_optimizers(
        self,
    ) -> Tuple[
        Callable[
            [bool],
            Union[Optimizer, List[Optimizer], List[LightningOptimizer]],
        ],
        List[Any],
    ]:
        optimizer = self.build_optimizer(
            params=[x for x in self.model.parameters() if x.requires_grad],
        )

        scheduler = self.build_scheduler(
            optimizer=optimizer,
        )

        self.optimizers = [optimizer]  # type: ignore
        return self.optimizers, [scheduler]  # type: ignore

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ):  # type: ignore
        images = batch["image"]
        targets = batch["annotation"]

        out = self.forward(images)

        loss_localization, loss_classification, loss_landmarks = self.loss(
            out,
            targets,
        )

        total_loss = (
            self.loss_weights["localization"] * loss_localization
            + self.loss_weights["classification"] * loss_classification
            + self.loss_weights["landmarks"] * loss_landmarks
        )

        self.log(
            "train_classification",
            loss_classification,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.log(
            "train_localization",
            loss_localization,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.log(
            "train_landmarks",
            loss_landmarks,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.log(
            "train_loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.log(
            "lr",
            self._get_current_lr(),
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        return total_loss

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ):  # type: ignore
        images = batch["image"]

        image_height = images.shape[2]
        image_width = images.shape[3]

        annotations = batch["annotation"]
        file_names = batch["file_name"]

        out = self.forward(images)

        location, confidence, _ = out

        confidence = F.softmax(confidence, dim=-1)
        batch_size = location.shape[0]

        predictions_coco: List[Dict[str, Any]] = []

        scale = torch.from_numpy(np.tile([image_width, image_height], 2)).to(
            location.device
        )

        for batch_id in range(batch_size):
            boxes = decode(
                location.data[batch_id],
                self.prior_box.to(images.device),
                self.config.test_parameters.variance,
            )
            scores = confidence[batch_id][:, 1]

            valid_index = torch.where(scores > 0.1)[0]
            boxes = boxes[valid_index]
            scores = scores[valid_index]

            boxes *= scale

            # do NMS
            keep = nms(boxes, scores, self.config.val_parameters.iou_threshold)
            boxes = boxes[keep, :].cpu().numpy()

            if boxes.shape[0] == 0:
                continue

            scores = scores[keep].cpu().numpy()

            file_name = file_names[batch_id]

            for box_id, bbox in enumerate(boxes):
                x_min, y_min, x_max, y_max = bbox

                x_min = np.clip(x_min, 0, x_max - 1)
                y_min = np.clip(y_min, 0, y_max - 1)

                predictions_coco += [
                    {
                        "id": str(hash(f"{file_name}_{box_id}")),
                        "image_id": file_name,
                        "category_id": 1,
                        "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                        "score": scores[box_id],
                    }
                ]

        gt_coco: List[Dict[str, Any]] = []

        for batch_id, annotation_list in enumerate(annotations):
            for annotation in annotation_list:
                x_min, y_min, x_max, y_max = annotation[:4]
                file_name = file_names[batch_id]

                gt_coco += [
                    {
                        "id": str(hash(f"{file_name}_{batch_id}")),
                        "image_id": file_name,
                        "category_id": 1,
                        "bbox": [
                            x_min.item() * image_width,
                            y_min.item() * image_height,
                            (x_max - x_min).item() * image_width,
                            (y_max - y_min).item() * image_height,
                        ],
                    }
                ]

        return OrderedDict({"predictions": predictions_coco, "gt": gt_coco})

    # TODO: Uncomment me
    # def validation_epoch_end(self, outputs: List) -> None:
    #     result_predictions: List[dict] = []
    #     result_gt: List[dict] = []

    #     for output in outputs:
    #         result_predictions += output["predictions"]
    #         result_gt += output["gt"]

    #     _, _, average_precision = recall_precision(
    #         result_gt,
    #         result_predictions,
    #         0.5,
    #     )

    #     self.log(
    #         "epoch",
    #         self.trainer.current_epoch,
    #         on_step=False,
    #         on_epoch=True,
    #         logger=True,
    #     )  # type: ignore
    #     self.log(
    #         "val_loss",
    #         average_precision,
    #         on_step=False,
    #         on_epoch=True,
    #         logger=True,
    #     )

    def _get_current_lr(self) -> torch.Tensor:  # type: ignore
        lr = [x["lr"] for x in self.optimizers[0].param_groups][0]  # type: ignore # noqa
        return torch.from_numpy(np.array([lr]))[0].to(self.device)
