from dataclasses import dataclass
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from wasp.retinaface.encode import encode
from wasp.retinaface.encode import encode_landm as encl
from wasp.retinaface.matching import match

T4 = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def log_sum_exp(x):
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


@dataclass
class LossWeights:
    localization: float
    classification: float
    landmarks: float
    depths: float


def masked_loss(
    loss_function,
    data: torch.Tensor,
    pred: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = ~torch.isnan(data)

    data_masked = data[mask]
    pred_masked = pred[mask]

    loss = loss_function(
        data_masked,
        pred_masked,
    )
    if data_masked.numel() == 0:
        loss = torch.nan_to_num(loss, 0)

    return loss, max(data_masked.shape[0], 1)


def depths_loss(
    positive: torch.Tensor,
    dpt_pred: torch.Tensor,
    dpths_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return masked_loss(
        partial(F.smooth_l1_loss, reduction="sum"),
        data=dpths_t[positive],
        pred=dpt_pred[positive],
    )


def localization_loss(
    positive: torch.Tensor,
    boxes_pred: torch.Tensor,
    boxes_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Localization Loss (Smooth L1) Shape: [batch, num_priors, 4]
    return masked_loss(
        partial(F.smooth_l1_loss, reduction="sum"),
        data=boxes_t[positive],
        pred=boxes_pred[positive],
    )


def landmark_loss(
    positive: torch.Tensor,
    kypts_pred: torch.Tensor,
    kypts_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return masked_loss(
        partial(F.smooth_l1_loss, reduction="sum"),
        data=kypts_t[positive],
        pred=kypts_pred[positive],
    )


def confidence_loss(
    positive: torch.Tensor,
    label_t: torch.Tensor,  # shape [n_batch, n_anchors]
    confidence_data: torch.Tensor,  # [n_batch, n_anchors, num_classes]
    neg: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    combined_mask = positive | neg

    # Use this combined mask to index confidence_data and label_t
    conf_p = confidence_data[combined_mask]
    targets_weighted = label_t[combined_mask].view(-1)

    # Compute confidenc loss
    loss_c = F.cross_entropy(
        conf_p.view(-1, num_classes),
        targets_weighted,
        reduction="sum",
    )

    # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
    num_pos = positive.long().sum(1, keepdim=True)
    n = max(num_pos.data.sum().float(), 1)  # type: ignore
    return loss_c, n


def mine_negatives(
    label,  # [batch, n_anchors]
    pred,
    n_batch,
    negpos_ratio,
    num_classes,
    positive,  # [batch, n_anchors]
):
    batch_conf = pred.view(-1, num_classes)
    loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, label.view(-1, 1))
    n_batch = label.shape[0]

    # Hard Negative Mining
    loss_c[positive.view(-1, 1)] = 0  # filter out positive boxes for now
    loss_c = loss_c.view(n_batch, -1)
    _, loss_idx = loss_c.sort(1, descending=True)
    _, idx_rank = loss_idx.sort(1)
    num_pos = positive.long().sum(1, keepdim=True)
    num_neg = torch.clamp(negpos_ratio * num_pos, max=positive.shape[1] - 1)
    return idx_rank < num_neg.expand_as(idx_rank)


class MultiBoxLoss(nn.Module):
    def __init__(
        self,
        priors: torch.Tensor,
        weights: LossWeights = LossWeights(
            localization=2,
            classification=1,
            landmarks=1,
            depths=4,
        ),
        num_classes: int = 2,
        overlap_thresh: float = 0.35,
        neg_pos: int = 7,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.negpos_ratio = neg_pos
        self.variance = [0.1, 0.2]
        self.priors = priors
        self.weights = weights

    def forward(self, predictions: T4, targets: torch.Tensor) -> T4:
        """Multibox Loss.

        Args:
            predictions: A tuple containing locations predictions, confidence,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size, num_priors, num_classes)
                loc shape: torch.size(batch_size, num_priors, 4)
                priors shape: torch.size(num_priors, 4)

            targets: Ground truth boxes and labels_gt for a batch,
                shape: [batch_size, num_objs, 5] (last box_index is the label).
        """
        boxes_pred, conf_pred, kpts_pred, dpth_pred = predictions

        device = targets[0]["boxes"].device

        priors = self.priors.to(device)

        n_batch = boxes_pred.shape[0]
        num_priors = priors.shape[0]

        # match priors (default boxes) and ground truth boxes
        label_t = torch.zeros(n_batch, num_priors).to(device).long()
        boxes_t = torch.zeros(n_batch, num_priors, 4).to(device)
        kypts_t = torch.zeros(n_batch, num_priors, 10).to(device)
        dpths_t = torch.zeros(n_batch, num_priors, 2).to(device)

        for i in range(n_batch):
            box_gt = targets[i]["boxes"]
            landmarks_gt = targets[i]["keypoints"]
            labels_gt = targets[i]["labels"]
            depths_gt = targets[i]["depths"]

            # matched gt index
            matched, labels = match(
                labels_gt.view(-1),
                box_gt,
                priors.data,
                self.threshold,
            )

            if matched is None:
                label_t[i] = 0
                boxes_t[i] = 0
                kypts_t[i] = 0
                dpths_t[i] = 0
                continue

            label_t[i] = labels  # [num_priors] top class label prior
            boxes_t[i] = encode(box_gt[matched], priors, self.variance)
            kypts_t[i] = encl(landmarks_gt[matched], priors, self.variance)
            dpths_t[i] = depths_gt[matched]

        positives = label_t != torch.zeros_like(label_t)
        positive = torch.where(positives)
        loss_landm, n1 = landmark_loss(positive, kpts_pred, kypts_t)
        loss_dpth, ndpth = depths_loss(positive, dpth_pred, dpths_t)
        loss_l, nl = localization_loss(positive, boxes_pred, boxes_t)

        label = label_t.detach().clone()
        label[positive] = 1

        negatives = mine_negatives(
            label=label,
            pred=conf_pred,
            n_batch=n_batch,
            negpos_ratio=self.negpos_ratio,
            num_classes=self.num_classes,
            positive=positives,
        )

        loss_c, n = confidence_loss(
            positives,
            label,
            conf_pred,
            negatives,
            self.num_classes,
        )

        return loss_l / nl, loss_c / n, loss_landm / n1, loss_dpth / ndpth

    def full_forward(
        self,
        predictions: T4,
        targets: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        localization, classification, landmarks, depths = self(
            predictions,
            targets,
        )

        total = (
            self.weights.localization * localization
            + self.weights.classification * classification
            + self.weights.landmarks * landmarks
            + self.weights.depths * depths
        )

        return total, localization, classification, landmarks, depths
