import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.ops import nms

import wasp.retinaface.augmentations as augs
from wasp.retinaface.data import FaceDetectionDataset, detection_collate
from wasp.retinaface.matching import decode
from wasp.retinaface.metrics import recall_precision


def dpath(envv):
    def f():
        Path(os.environ[envv])

    return f


@dataclass
class Paths:
    train: Path = field(default_factory=dpath("TRAIN_LABEL_PATH"))
    valid: Path = field(default_factory=dpath("VAL_LABEL_PATH"))


def prepare_outputs(
    images,
    out,
    targets,
    prior_box,
) -> list[tuple[np.ndarray, np.ndarray]]:
    image_height = images.shape[2]
    image_width = images.shape[3]

    location, confidence, _ = out

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
        boxes = boxes[valid_index]
        scores = scores[valid_index]

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

        tt = target.cpu().numpy()
        gts = np.zeros((target.shape[0], 7), dtype=np.float32)
        gts[:, :4] = tt[:, :4] * scale[None, :].cpu().numpy()
        gts[:, 4] = np.where(tt[:, -1] > 0, 0, 1)
        total.append((candidates, gts))

    return total


class RetinaFacePipeline(pl.LightningModule):  # pylint: disable=R0901
    def __init__(
        self,
        paths: Paths,
        model: torch.nn.Module,
        preprocessing,
        priorbox,
        build_optimizer,
        build_scheduler,
        loss,
    ) -> None:
        super().__init__()
        self.paths = paths
        self.model = model
        self.prior_box = priorbox
        self.loss = loss
        self.preproc = preprocessing
        self.build_optimizer = build_optimizer
        self.build_scheduler = build_scheduler
        self.validation_outputs: list[dict] = []

    def forward(
        self, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore
        return self.model(batch)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            FaceDetectionDataset(
                label_path=self.paths.train,
                transform=augs.train(),
                preproc=self.preproc,
                rotate90=True,
            ),
            batch_size=8,
            num_workers=4,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            collate_fn=detection_collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            FaceDetectionDataset(
                label_path=self.paths.valid,
                transform=augs.valid(),
                preproc=self.preproc,
                rotate90=True,
            ),
            batch_size=10,
            num_workers=4,
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
        total_loss, loss_loc, loss_clf, loss_lmrks = self.loss.full_forward(
            out,
            targets,
        )

        self.log(
            "train_classification",
            loss_clf,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.log(
            "train_localization",
            loss_loc,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.log(
            "train_landmarks",
            loss_lmrks,
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
        out = self.forward(images)

        outputs = prepare_outputs(
            images=images,
            out=out,
            targets=batch["annotation"],
            prior_box=self.prior_box,
        )
        self.validation_outputs.append(outputs)
        return outputs

    def on_validation_epoch_end(self) -> None:
        result_predictions: List[dict] = []
        result_gt: List[dict] = []

        for output in self.validation_outputs:
            result_predictions += output["predictions"]
            result_gt += output["gt"]

        _, _, average_precision = recall_precision(
            result_gt,
            result_predictions,
            0.5,
        )

        self.log(
            "epoch",
            self.trainer.current_epoch,
            on_step=False,
            on_epoch=True,
            logger=True,
        )  # type: ignore
        self.log(
            "val_loss",
            average_precision,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.validation_outputs = []

    def _get_current_lr(self) -> torch.Tensor:  # type: ignore
        lr = [x["lr"] for x in self.optimizers[0].param_groups][0]  # type: ignore # noqa
        return torch.from_numpy(np.array([lr]))[0].to(self.device)
