import warnings
from typing import Any, Callable, Dict, List, Tuple, Union

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
from wasp.retinaface.data import FaceDetectionDataset, detection_collate
from wasp.retinaface.encode import decode
from wasp.retinaface.preprocess import normalize


def prepare_outputs(
    images,
    out,
    targets,
    prior_box,
) -> list[tuple[np.ndarray, np.ndarray]]:
    image_height = images.shape[2]
    image_width = images.shape[3]

    location, confidence, *_ = out

    confidence = F.softmax(confidence, dim=-1)
    scale = torch.from_numpy(np.tile([image_width, image_height], 2)).to(
        location.device
    )

    total = []
    for batch_id, target in enumerate(targets):
        boxes = decode(
            location.data[batch_id],
            prior_box.to(images.device),
            [0.1, 0.2],
        )
        scores = confidence[batch_id][:, 1]

        valid_index = torch.where(scores > 0.1)[0]
        # NMS doesn't accept fp16 inputs
        boxes = boxes[valid_index].float()
        scores = scores[valid_index].float()

        boxes *= scale

        # do NMS
        keep = nms(boxes, scores, 0.4)
        boxes = boxes[keep, :].cpu().numpy()
        scores = scores[keep].cpu().numpy()
        candidates = np.concatenate(
            (boxes, scores.reshape(-1, 1), scores.reshape(-1, 1)),
            axis=1,
        )
        candidates[:, -2] = 0

        boxes_gt = target["boxes"].cpu().numpy()
        labels_gt = target["labels"].cpu().numpy()
        gts = np.zeros((boxes_gt.shape[0], 7), dtype=np.float32)
        gts[:, :4] = boxes_gt[:, :4] * scale[None, :].cpu().numpy()
        gts[:, 4] = np.where(labels_gt[:, -1] > 0, 0, 1)
        total.append((candidates, gts))

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
            num_classes=1,
        )
        self.prepare_outputs = prepare_outputs

    def forward(
        self, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore
        return self.model(batch)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            FaceDetectionDataset(
                label_path=self.train_labels,
                transform=augs.train(self.resolution),
                preproc=normalize,
                rotate90=False,
            ),
            batch_size=32,
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
            ),
            batch_size=32,
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
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ):  # type: ignore
        images = batch["image"]
        targets = batch["annotation"]
        # Don't provide images
        # targets[0]["images"] = images

        out = self.forward(images)
        losses = self.loss(
            out,
            targets,
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
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ):  # type: ignore
        images = batch["image"]
        targets = batch["annotation"]
        targets[0]["images"] = images

        out = self.forward(images)
        losses = self.loss(
            out,
            targets,
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
            out=out,
            targets=batch["annotation"],
            prior_box=self.prior_box,
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
