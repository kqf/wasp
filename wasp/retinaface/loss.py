from dataclasses import dataclass
from functools import partial
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from wasp.retinaface.data import DetectionTargets
from wasp.retinaface.encode import encode
from wasp.retinaface.matching import match2

T4 = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


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


def mine_negatives(
    label: torch.Tensor,
    pred: torch.Tensor,
    negpos_ratio: int,
    positive: torch.Tensor,
):
    batch_size, num_anchors, _ = positive.shape
    pos_batch, pos_anchor, pos_obj = torch.where(positive)
    labels = torch.zeros_like(pred[:, :, 0], dtype=torch.long)
    labels[pos_batch, pos_anchor] = label[pos_batch, pos_obj].squeeze()
    loss = F.cross_entropy(
        pred.view(-1, pred.shape[-1]), labels.view(-1), reduction="none"
    ).view(batch_size, num_anchors)
    loss[pos_batch, pos_anchor] = 0
    _, loss_sorted_idx = loss.sort(dim=1, descending=True)
    _, rank = loss_sorted_idx.sort(dim=1)
    num_pos = positive.sum(dim=(1, 2), dtype=torch.long).unsqueeze(1)
    num_neg = torch.clamp(negpos_ratio * num_pos, max=num_anchors - 1)
    return rank < num_neg.expand_as(rank)


@torch.no_grad()
def stack(tensors, pad_value=0) -> torch.Tensor:
    max_length = max(tensor.shape[0] for tensor in tensors)
    return torch.stack(
        [
            F.pad(t, (0, 0, 0, max_length - t.shape[0]), value=pad_value)
            for t in tensors
        ]
    )


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


def match_combined(
    classes,
    boxes,
    priors,
    confidences,
    negpos_ratio,
    threshold,
):
    positives = torch.stack(
        [match2(c, b, priors, threshold) for c, b in zip(classes, boxes)]
    )
    negatives = mine_negatives(classes, confidences, negpos_ratio, positives)
    return positives, negatives


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


def default_loss(num_classes, variances) -> dict[str, WeightedLoss]:
    return {
        "classes": WeightedLoss(
            partial(confidence_loss, num_classes=num_classes),
            2,
            needs_negatives=True,
        ),
        "boxes": WeightedLoss(
            localization_loss,
            1,
            enc_true=partial(encode, variances=variances),
            needs_negatives=False,
        ),
    }


class MultiBoxLoss(nn.Module):
    def __init__(
        self,
        priors: torch.Tensor,
        sublosses: dict[str, WeightedLoss] = None,
        num_classes: int = 2,
        overlap_thresh: float = 0.35,
        neg_pos: int = 7,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.negpos_ratio = neg_pos
        self.variance = [0.1, 0.2]
        self.sublosses = sublosses or default_loss(num_classes, self.variance)
        self.register_buffer("priors", priors)

    def forward(
        self,
        y_pred: DetectionTargets,
        y_true: DetectionTargets,
    ) -> dict[str, torch.Tensor]:
        positives, negatives = match_combined(
            y_true.classes,
            y_true.boxes,
            self.priors,
            confidences=y_pred.classes,
            negpos_ratio=self.negpos_ratio,
            threshold=self.threshold,
        )

        losses = {}
        for name, subloss in self.sublosses.items():
            y_pred_, y_true_, anchor_ = select(
                y_pred.__dict__[name],
                y_true.__dict__[name],
                self.priors,
                use_negatives=subloss.needs_negatives,
                positives=positives,
                negatives=negatives,
            )
            losses[name] = subloss(y_pred_, y_true_, anchor_)

        losses["loss"] = torch.stack(tuple(losses.values())).sum()
        return losses
