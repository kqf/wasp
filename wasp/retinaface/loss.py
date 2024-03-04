from dataclasses import dataclass
from functools import partial
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from wasp.retinaface.encode import encode
from wasp.retinaface.encode import encode_landm as encl
from wasp.retinaface.matching import match


def log_sum_exp(x):
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


@dataclass
class LossWeights:
    localization: float
    classification: float
    landmarks: float


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


class MultiBoxLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        overlap_thresh: float,
        prior_for_matching: bool,
        neg_mining: bool,
        neg_pos: int,
        neg_overlap: float,
        encode_target: bool,
        priors: torch.Tensor,
        weights: LossWeights,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]
        self.priors = priors
        self.weights = weights

    def forward(
        self,
        predictions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        locations_data, confidence_data, landmark_data = predictions
        device = targets[0]["boxes"].device

        priors = self.priors.to(device)
        n_predictions = locations_data.shape[0]
        num_priors = priors.shape[0]

        # match priors (default boxes) and ground truth boxes
        label_t = torch.zeros(n_predictions, num_priors).to(device).long()
        boxes_t = torch.zeros(n_predictions, num_priors, 4).to(device)
        kypts_t = torch.zeros(n_predictions, num_priors, 10).to(device)
        dpths_t = torch.zeros(n_predictions, num_priors, 2).to(device)

        for i in range(n_predictions):
            box_gt = targets[i]["boxes"].data
            landmarks_gt = targets[i]["keypoints"].data
            labels_gt = targets[i]["labels"].reshape(-1).data
            depths_gt = targets[i]["depths"].data

            # matched gt index
            matched, labels = match(
                labels_gt,
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

        # landmark Loss (Smooth L1) Shape: [batch, num_priors, 10]
        positive_1 = label_t > torch.zeros_like(label_t)
        num_positive_landmarks = positive_1.long().sum(1, keepdim=True)
        n1 = max(num_positive_landmarks.data.sum().float(), 1)  # type: ignore
        pos_idx1 = positive_1.unsqueeze(positive_1.dim()).expand_as(
            landmark_data,
        )

        loss_landm, n1 = masked_loss(
            partial(F.smooth_l1_loss, reduction="sum"),
            data=kypts_t[pos_idx1].view(-1, 10),
            pred=landmark_data[pos_idx1].view(-1, 10),
        )

        positive = label_t != torch.zeros_like(label_t)
        label_t[positive] = 1

        # Localization Loss (Smooth L1) Shape: [batch, num_priors, 4]
        pos_idx = positive.unsqueeze(positive.dim()).expand_as(locations_data)
        loc_p = locations_data[pos_idx].view(-1, 4)
        boxes_t = boxes_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, boxes_t, reduction="sum")

        # Compute max conf across batch for hard negative mining
        batch_conf = confidence_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(
            1, label_t.view(-1, 1)
        )  # noqa

        # Hard Negative Mining
        loss_c[positive.view(-1, 1)] = 0  # filter out positive boxes for now
        loss_c = loss_c.view(n_predictions, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = positive.long().sum(1, keepdim=True)
        num_neg = torch.clamp(
            self.negpos_ratio * num_pos, max=positive.shape[1] - 1
        )  # noqa
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = positive.unsqueeze(2).expand_as(confidence_data)
        neg_idx = neg.unsqueeze(2).expand_as(confidence_data)
        conf_p = confidence_data[(pos_idx + neg_idx).gt(0)].view(
            -1, self.num_classes
        )  # noqa
        targets_weighted = label_t[(positive + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction="sum")

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        n = max(num_pos.data.sum().float(), 1)  # type: ignore

        return loss_l / n, loss_c / n, loss_landm / n1

    def full_forward(
        self,
        predictions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        localization, classification, landmarks = self(
            predictions,
            targets,
        )

        total = (
            self.weights.localization * localization
            + self.weights.classification * classification
            + self.weights.landmarks * landmarks
        )

        return total, localization, classification, landmarks
