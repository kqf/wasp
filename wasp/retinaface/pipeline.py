import warnings
from typing import Any, Callable, List, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from mean_average_precision import MetricBuilder
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.ops import nms

import wasp.retinaface.augmentations as augs
from wasp.retinaface.data import (
    Batch,
    DetectionTask,
    FaceDetectionDataset,
    detection_collate,
)
from wasp.retinaface.encode import decode
from wasp.retinaface.preprocess import normalize


def prepare_outputs(
    images: torch.Tensor,
    y_pred: DetectionTask,
    y_true: DetectionTask,
    anchors: torch.Tensor,
) -> list[tuple[np.ndarray, np.ndarray]]:
    image_w = images.shape[2]
    image_h = images.shape[3]

    confidence = F.softmax(y_pred.classes, dim=-1)
    scale = torch.from_numpy(
        np.tile(
            [image_h, image_w],
            2,
        )
    ).to(y_pred.boxes.device)

    total = []
    batch = zip(y_true.classes, y_true.boxes)
    for batch_id, (y_true_label, y_true_boxes) in enumerate(batch):
        boxes_pred = decode(
            y_pred.boxes[batch_id],
            anchors.to(images.device),
            [0.1, 0.2],
        )
        # NB: it's desired to start class_ids from 0,
        # 0 is for background it's not included
        scores = confidence[batch_id][:, 1:]

        valid_index = torch.where((scores > 0.1).any(-1))[0]
        # NMS doesn't accept fp16 inputs
        boxes_pred = boxes_pred[valid_index].float()
        scores = scores[valid_index].float()
        probs_pred, label_pred = scores.max(dim=-1)

        boxes_pred *= scale

        # do NMS
        keep = nms(boxes_pred, probs_pred, 0.4)
        boxes_pred = boxes_pred[keep, :].cpu().numpy()
        probs_pred = probs_pred[keep].cpu().numpy()
        label_pred = label_pred[keep].cpu().numpy()
        pred = np.concatenate(
            (
                boxes_pred,
                label_pred.reshape(-1, 1),
                probs_pred.reshape(-1, 1),
            ),
            axis=1,
        )

        boxes_true = y_true_boxes.cpu().numpy()
        label_true = y_true_label.cpu().numpy()
        true = np.zeros((boxes_true.shape[0], 7), dtype=np.float32)
        true[:, :4] = boxes_true[:, :4] * scale[None, :].cpu().numpy()
        # While calculating mAP, always start with 0
        # We don't calculate the metrics for background class
        true[:, 4] = label_true[:, -1] - 1
        total.append((pred, true))
    return total


class RetinaFacePipeline(pl.LightningModule):  # pylint: disable=R0901
    def __init__(
        self,
        train_labels: str,
        valid_labels: str,
        model: torch.nn.Module,
        resolution: tuple[int, int],
        preprocessing,
        priorbox,
        build_optimizer,
        build_scheduler,
        loss,
        mapping: dict[str, int],
        prepare_outputs=prepare_outputs,
    ) -> None:
        super().__init__()
        self.train_labels = train_labels
        self.valid_labels = valid_labels
        self.model = model
        self.resolution = resolution
        self.prior_box = priorbox
        self.loss = loss
        self.preproc = preprocessing
        self.build_optimizer = build_optimizer
        self.build_scheduler = build_scheduler
        self.metric_fn = MetricBuilder.build_evaluation_metric(
            "map_2d",
            async_mode=False,
            num_classes=loss.num_classes - 1,
        )
        self.prepare_outputs = prepare_outputs
        self.mapping = mapping

    def forward(
        self, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore
        return self.model(batch)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            FaceDetectionDataset(
                label_path=self.train_labels,
                transform=augs.valid(self.resolution),
                preproc=normalize,
                rotate90=False,
                mapping=self.mapping,
            ),
            batch_size=4,
            num_workers=12,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            collate_fn=detection_collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            FaceDetectionDataset(
                label_path=self.valid_labels,
                transform=augs.valid(self.resolution),
                preproc=normalize,
                rotate90=False,
                mapping=self.mapping,
            ),
            batch_size=4,
            num_workers=12,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
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
        batch: Batch,
        batch_idx: int,
    ):  # type: ignore
        images = batch.image
        y_true = batch.annotation
        # Don't provide images
        # targets[0]["images"] = images

        y_pred = self.forward(images)
        losses = self.loss(
            y_pred,
            y_true,
        )

        for name, loss in losses.items():
            self.log(
                f"train_{name}",
                loss,
                on_step=True,
                on_epoch=True,
                logger=True,
                prog_bar=True,
                sync_dist=True,
                batch_size=images.shape[0],
            )

        self.log(
            "lr",
            self._get_current_lr(),
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=images.shape[0],
        )

        return losses["loss"]

    def validation_step(
        self,
        batch: Batch,
        batch_idx: int,
    ):  # type: ignore
        images = batch.image
        y_true = batch.annotation
        # y_true[0]["images"] = images

        y_pred = self.forward(images)
        losses = self.loss(
            y_pred,
            y_true,
        )

        for name, loss in losses.items():
            self.log(
                f"valid_{name}",
                loss,
                on_step=True,
                on_epoch=True,
                logger=True,
                prog_bar=True,
                sync_dist=True,
                batch_size=images.shape[0],
            )

        outputs = self.prepare_outputs(
            images=images,
            y_pred=y_pred,
            y_true=batch.annotation,
            anchors=self.prior_box,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for perimage in outputs:
                self.metric_fn.add(*perimage)
        return batch

    def on_validation_epoch_end(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            average_precision = self.metric_fn.value(iou_thresholds=0.5)["mAP"]

        self.metric_fn.reset()
        self.log(
            "epoch",
            self.trainer.current_epoch,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )  # type: ignore
        self.log(
            "mAP",
            average_precision,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

    def _get_current_lr(self) -> torch.Tensor:  # type: ignore
        lr = [x["lr"] for x in self.optimizers[0].param_groups][0]  # type: ignore # noqa
        return torch.from_numpy(np.array([lr]))[0].to(self.device)
