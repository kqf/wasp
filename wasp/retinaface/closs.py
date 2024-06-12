from dataclasses import dataclass
from functools import partial
from typing import Callable

import torch
import torchvision

from wasp.retinaface.encode import encode
from wasp.retinaface.encode import encode_landm as encl
from wasp.retinaface.matching import iou


def match_positives(score, pos_th):
    # socre[batch_size, n_obj, n_anchor]
    max_overlap = torch.abs(score.max(dim=1, keepdim=True)[0] - score) < 1.0e-6
    positive = max_overlap & (score > pos_th)
    return positive


def match(
    boxes,  # [batch_size, n_obj, 4]
    mask,  # [batch_size, n_obj]
    anchors,  # [batch_size, n_anchors, 4]
    on_image=None,  # [batch_size, n_anchors]
    criterion=iou,
    pos_th=0.5,
    neg_th=0.1,
    fill_value=-1,
):
    # criterion([batch_size, 1, n_anchors, 4], [batch_size, n_obj, 1, 4])
    # ~> overlap[batch_size, n_obj, n_anchor]
    overlap = criterion(
        anchors[:, None],
        boxes[:, :, None],
    )

    # Remove all scores that are masked
    overlap[mask] = fill_value

    positive = match_positives(overlap, pos_th)

    # Check if within image
    if on_image is not None:
        positive = positive & on_image[..., None].bool()

    # Negatives are the anchors that have quite small
    # largest overlap with objects
    # overlap[batch_size, n_obj, n_anchor]
    overlap_, _ = overlap.max(dim=1)
    negative = overlap_ < neg_th

    return positive, negative, overlap


def select(y_pred, y_true, anchor, positives, negatives, use_negatives=True):
    batch_, obj_, anchor_ = torch.where(positives)
    y_pred_pos = y_pred[batch_, anchor_]
    y_true_pos = y_true[batch_, obj_]
    anchor_pos = anchor[batch_, anchor_]

    if not use_negatives:
        return y_pred_pos, y_true_pos, anchor_pos

    y_pred_neg = y_pred[torch.where(negatives)]
    anchor_neg = anchor[torch.where(negatives)]

    # Zero is a background
    y_true_neg_shape = [y_pred_neg.shape[0]]
    if len(y_true_pos.shape) > 1:
        y_true_neg_shape.append(y_true_pos.shape[-1])

    # Assume that zero is the negative class, increase the labels by 1
    y_true_neg = torch.zeros(y_true_neg_shape, device=y_true_pos.device)

    y_pred_tot = torch.cat([y_pred_pos, y_pred_neg], dim=0)
    anchor_tot = torch.cat([anchor_pos, anchor_neg], dim=0)
    # Increase y_true_pos by 1 since negatives are zeros
    y_true_tot = torch.squeeze(torch.cat([y_true_pos, y_true_neg], dim=0))
    return y_pred_tot, y_true_tot, anchor_tot


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


def masked_loss(
    pred: torch.Tensor,
    data: torch.Tensor,
    loss_function,
) -> torch.Tensor:
    if data.numel() == 0:
        return torch.tensor(0.0, device=data.device)

    mask = ~torch.isnan(data)

    data_masked = data[mask]
    pred_masked = pred[mask]

    loss = loss_function(
        data_masked,
        pred_masked,
    )
    if data_masked.numel() == 0:
        loss = torch.nan_to_num(loss, 0)
    return loss / max(data_masked.shape[0], 1)


def default_losses(variance=None):
    variance = variance or [0.1, 0.2]

    return {
        "boxes": WeightedLoss(
            partial(
                masked_loss,
                loss_function=torch.nn.SmoothL1Loss(),
            ),
            enc_true=lambda x, a: encode(x, a, variances=variance),
            weight=1,
        ),
        "keypoints": WeightedLoss(
            partial(
                masked_loss,
                loss_function=torch.nn.SmoothL1Loss(),
            ),
            enc_true=lambda x, a: encl(x, a, variances=variance),
            # enc_true=encode,
            weight=1,
        ),
        "depths": WeightedLoss(
            partial(
                masked_loss,
                loss_function=torch.nn.SmoothL1Loss(),
            ),
            weight=1,
        ),
        "classes": WeightedLoss(
            partial(
                masked_loss,
                loss_function=partial(
                    torchvision.ops.sigmoid_focal_loss,
                    reduction="mean",
                    alpha=0.8,
                    gamma=0.5,
                ),
            ),
            enc_true=lambda y, _: torch.nn.functional.one_hot(
                y.reshape(-1).long(), num_classes=2
            )
            .float()
            .clamp(0, 1.0),
            # enc_true=debug,
            needs_negatives=True,
        ),
    }


def stack(tensors, pad_value=-1) -> torch.Tensor:
    max_length = max(tensor.shape[0] for tensor in tensors)

    # Pad each tensor to the maximum length
    padded = []
    for t in tensors:
        # (left, right, top, bottom)
        padding = (0, 0, 0, max_length - t.shape[0])
        padded.append(torch.nn.functional.pad(t, padding, value=pad_value))

    return torch.stack(padded)


class DetectionLoss(torch.nn.Module):
    def __init__(self, sublosses=None, anchors=None):
        super().__init__()
        self.sublosses = sublosses or default_losses()
        # We need to register the losses to manage things properly
        self.registered = torch.nn.ModuleList(
            [
                loss.loss
                for loss in self.sublosses.values()
                if isinstance(loss.loss, torch.nn.Module)
            ]
        )
        self.register_buffer("anchors", anchors[None])

    def forward(self, predictions, targets):
        y = {
            "classes": stack([target["labels"] for target in targets]),
            "boxes": stack([target["boxes"] for target in targets]),
            "keypoints": stack(
                [target["keypoints"] for target in targets],
            ),
            "depths": stack([target["depths"] for target in targets]),
        }
        y_pred = {
            "classes": predictions[1],
            "boxes": predictions[0],
            "keypoints": predictions[2],
            "depths": predictions[3],
        }

        positives, negatives, _ = match(
            y["boxes"],
            (y["classes"] < 0).squeeze(-1),
            self.anchors,
        )

        # fselect -- selects only matched positives / negatives
        fselect = partial(select, positives=positives, negatives=negatives)
        losses = {}
        for name, subloss in self.sublosses.items():
            # fselect(
            #   y_pred[batch, n_detections, dim1],
            #   y_true[batch, n_objects, dim2],
            #   anchor[batch, n_detections, 4],
            # )
            # ~> y_pred_[n_samples, dim1]
            # ~> y_true_[n_samples, dim2]
            # ~> anchor_[n_samples, 4]

            y_pred_, y_true_, anchor_ = fselect(
                y_pred[name],
                y[name],
                self.anchors,
                use_negatives=subloss.needs_negatives,
            )
            losses[name] = subloss(y_pred_, y_true_, anchor_)

        losses["loss"] = torch.stack(tuple(losses.values())).sum()
        return losses
