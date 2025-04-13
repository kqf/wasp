from dataclasses import fields
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from wasp.retinaface.data import DetectionTargets, WeightedLoss
from wasp.retinaface.encode import encode
from wasp.retinaface.matching import match


def masked_loss(
    loss_function,
    pred: torch.Tensor,
    data: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = ~torch.isnan(data)
    data_masked = data[mask]
    pred_masked = pred[mask]
    loss = loss_function(data_masked, pred_masked)
    if data_masked.numel() == 0:
        loss = torch.nan_to_num(loss, 0)
    return loss / max(data_masked.shape[0], 1)


def localization_loss(
    pred: torch.Tensor,
    data: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return masked_loss(
        partial(F.smooth_l1_loss, reduction="sum"),
        pred,
        data,
    )


def confidence_loss(
    pred: torch.Tensor,
    data: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Calculate number of positive examples as all non zero labels
    n_pos = (data > 0).sum()
    loss_c = F.cross_entropy(
        pred.reshape(-1, num_classes),
        data.view(-1),
        reduction="sum",
    )
    return loss_c / n_pos


def select(y_pred, y_true, anchors, use_negatives, positives, negatives):
    b, a, o = torch.where(positives)
    if not use_negatives:
        return y_pred[b, a], y_true[b, o], anchors[a]

    # TODO: Fix this logic
    conf_pos = y_pred[b, a]
    targets_pos = y_true[b, o].view(-1)
    b, a = torch.where(negatives)
    conf_neg = y_pred[b, a]
    targets_neg = torch.zeros_like(conf_neg[:, 0], dtype=torch.long)
    conf_all = torch.cat([conf_pos, conf_neg], dim=0)
    targets_all = torch.cat([targets_pos, targets_neg], dim=0).long()
    return conf_all, targets_all, anchors[a]


def default_loss(
    num_classes,
    variances,
) -> DetectionTargets[Optional[WeightedLoss]]:
    return DetectionTargets(
        classes=WeightedLoss(
            partial(confidence_loss, num_classes=num_classes),
            weight=2,
            needs_negatives=True,
        ),
        boxes=WeightedLoss(
            localization_loss,
            weight=1,
            enc_true=partial(encode, variances=variances),
            needs_negatives=False,
        ),
        keypoints=None,
        depths=None,
    )


class MultiBoxLoss(nn.Module):
    def __init__(
        self,
        priors: torch.Tensor,
        sublosses: DetectionTargets[Optional[WeightedLoss]] = None,
        num_classes: int = 2,
        matching_overlap: float = 0.35,
        neg_pos: int = 7,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.matching_overlap = matching_overlap
        self.negpos_ratio = neg_pos
        self.variance = [0.1, 0.2]
        self.sublosses = sublosses or default_loss(num_classes, self.variance)
        self.register_buffer("priors", priors)

    def forward(
        self,
        y_pred: DetectionTargets,
        y_true: DetectionTargets,
    ) -> dict[str, torch.Tensor]:
        positives, negatives = match(
            y_true.classes,
            y_true.boxes,
            self.priors,
            confidences=y_pred.classes,
            negpos_ratio=self.negpos_ratio,
            overalp=self.matching_overlap,
        )

        losses = {}
        for field in fields(self.sublosses):
            name = field.name
            subloss: Optional[WeightedLoss] = getattr(self.sublosses, name)
            if subloss is None:
                continue

            y_pred_, y_true_, anchor_ = select(
                getattr(y_pred, name),
                getattr(y_true, name),
                self.priors,
                use_negatives=subloss.needs_negatives,
                positives=positives,
                negatives=negatives,
            )
            losses[name] = subloss(y_pred_, y_true_, anchor_)

        losses["loss"] = torch.stack(tuple(losses.values())).sum()
        return losses
