import os
from dataclasses import dataclass
from functools import partial
from typing import Callable

import cv2
import numpy as np
import torch

from wasp.retinaface.encode import encode, point_form

# from wasp.retinaface.encode import encode_landm as encl
from wasp.retinaface.matching import iou

# import torchvision


def match_positives(score: torch.Tensor, pos_th: float) -> torch.Tensor:
    # socre[batch_size, n_obj, n_anchor]
    max_overlap = torch.abs(score.max(dim=1, keepdim=True)[0] - score) < 1.0e-6
    return max_overlap & (score > pos_th)


def match(
    boxes: torch.Tensor,  # [batch_size, n_obj, 4]
    mask: torch.Tensor,  # [batch_size, n_obj]
    anchors: torch.Tensor,  # [batch_size, n_anchors, 4]
    on_image=None,  # [batch_size, n_anchors]
    criterion: Callable = iou,
    pos_th: float = 0.5,
    neg_th: float = 0.5,
    fill_value: int = -1,
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

    return positive, negative


def mine_negatives(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    anchors: torch.Tensor,
    n_positive: int,
    neg_pos_ratio: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Compute the classification loss using cross_entropy
    loss = torch.nn.functional.cross_entropy(
        y_pred,
        y_true,
        reduction="none",
    )
    # Mask out first n_positive entries (those are positives)
    loss[:n_positive] = -float("inf")

    # Find 10 times more negatives
    _, hard_indices = torch.topk(
        loss,
        min(n_positive * neg_pos_ratio, y_pred.shape[0] - n_positive),
    )

    # Merge hard-negatives and positive indices
    total = torch.cat(
        (
            torch.arange(n_positive, device=y_pred.device),
            hard_indices,
        )
    )

    return y_pred[total], y_true[total], anchors[total]


def select(
    y_pred,
    y_true,
    anchor,
    positives,
    negatives,
    use_negatives=True,
    # mine_negatives=lambda x, y, n_pos: (x, y),
    # mine_negatives=mine_negatives,
    mine_negatives=lambda x, y, z, w, *args: (x, y, z, w),
):
    batch_, obj_, anchor_ = torch.where(positives)
    y_pred_pos = y_pred[batch_, anchor_]
    y_true_pos = y_true[batch_, obj_]
    anchor_pos = anchor[torch.zeros_like(batch_), anchor_]
    n_pos = y_pred_pos.shape[0]

    if not use_negatives:
        return y_pred_pos, y_true_pos, anchor_pos, n_pos

    neg_batch_, neg_obj_ = torch.where(negatives)
    y_pred_neg = y_pred[neg_batch_, neg_obj_]
    anchor_neg = anchor[torch.zeros_like(neg_batch_), neg_obj_]

    # Zero is a background
    y_true_neg_shape = [y_pred_neg.shape[0]]
    if len(y_true_pos.shape) > 1:
        y_true_neg_shape.append(y_true_pos.shape[-1])

    # Assume that zero is the negative class
    y_true_neg = torch.zeros(
        y_true_neg_shape,
        device=y_true_pos.device,
        dtype=y_true_pos.dtype,
    )
    y_pred_tot = torch.cat([y_pred_pos, y_pred_neg], dim=0)
    anchor_tot = torch.cat([anchor_pos, anchor_neg], dim=0)
    # Increase y_true_pos by 1 since negatives are zeros
    y_true_tot = torch.squeeze(torch.cat([y_true_pos, y_true_neg], dim=0))

    return mine_negatives(
        y_pred_tot,
        y_true_tot,
        anchor_tot,
        y_pred_pos.shape[0],
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


def masked_loss(
    pred: torch.Tensor,
    data: torch.Tensor,
    loss_function,
) -> torch.Tensor:
    if data.numel() == 0:
        return torch.tensor(
            0.0, device=data.device, requires_grad=True
        )  # Ensure gradient tracking

    mask = ~torch.isnan(data)

    try:
        data_masked = data[mask]
        pred_masked = pred[mask]
    except RuntimeError as e:
        print(f"===> {pred.shape=}, {data.shape=}, {mask.shape=}")
        raise e

    loss = loss_function(
        pred_masked,
        data_masked,
    )
    if data_masked.numel() == 0:
        return torch.tensor(
            0.0, device=data.device, requires_grad=True
        )  # Ensure gradient tracking

    # Check for non-finite values and return zero if any are found
    if not torch.isfinite(loss).all():
        print(f"Non-finite loss detected: {loss.item()}")
        return torch.tensor(
            0.0, device=data.device, requires_grad=True
        )  # Ensure gradient tracking

    return torch.nan_to_num(loss)


def default_losses(variance=None):
    variance = variance or [0.1, 0.2]

    return {
        "boxes": WeightedLoss(
            partial(
                masked_loss,
                loss_function=torch.nn.SmoothL1Loss(reduction="sum"),
            ),
            enc_true=lambda x, a: encode(x, a, variances=variance),
            weight=1,
        ),
        # "keypoints": WeightedLoss(
        #     partial(
        #         masked_loss,
        #         loss_function=torch.nn.SmoothL1Loss(),
        #     ),
        #     # enc_true=lambda x, a: encl(x, a, variances=variance),
        #     enc_true=encode,
        #     weight=1,
        # ),
        # "depths": WeightedLoss(
        #     partial(
        #         masked_loss,
        #         loss_function=torch.nn.SmoothL1Loss(),
        #     ),
        #     weight=1,
        # ),
        "classes": WeightedLoss(
            # partial(
            #     masked_loss,
            #     loss_function=partial(
            #         torchvision.ops.sigmoid_focal_loss,
            #         reduction="mean",
            #         alpha=0.8,
            #         gamma=0.5,
            #     ),
            # ),
            # enc_true=lambda y, _: torch.nn.functional.one_hot(
            #     y.reshape(-1).long(), num_classes=2
            # )
            # .float()
            # .clamp(0, 1.0),
            partial(
                masked_loss,
                loss_function=torch.nn.CrossEntropyLoss(
                    reduce="sum",
                    weight=torch.tensor([1.0, 4.0]),
                ),
            ),
            # enc_true=debug,
            needs_negatives=True,
            weight=10.0,
        ),
    }


@torch.no_grad()
def stack(tensors, pad_value=-1) -> torch.Tensor:
    max_length = max(tensor.shape[0] for tensor in tensors)

    # Pad each tensor to the maximum length
    padded = []
    for t in tensors:
        # (left, right, top, bottom)
        padding = (0, 0, 0, max_length - t.shape[0])
        padded.append(torch.nn.functional.pad(t, padding, value=pad_value))

    return torch.stack(padded)


def denormalize(image):
    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        dtype=image.dtype,
        device=image.device,
    )
    std = torch.tensor(
        [0.229, 0.224, 0.225],
        dtype=image.dtype,
        device=image.device,
    )

    # Reverse normalization: x' = (x * std) + mean
    image = (image * std[:, None, None]) + mean[:, None, None]
    image = image.clamp(0, 1)  # Clamping to [0, 1] range
    return image * 255.0


def draw_anchors_on_image(image, anchors, y_true_, y_pred_, odir):
    image = denormalize(image).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    anchors = anchors.cpu().numpy()

    # Convert image from torch (C, H, W) to OpenCV (H, W, C) and BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    height, width, _ = image.shape

    def rectangle(image, box, color):
        pt1 = (int(box[0] * width), int(box[1] * height))
        pt2 = (int(box[2] * width), int(box[3] * height))
        return cv2.rectangle(image, pt1, pt2, color, 2)

    # Draw each anchor box on the image
    for i, box in enumerate(anchors):
        image = rectangle(image, box, color=(255, 255, 255))

    for i, box in enumerate(y_true_):
        image = rectangle(image, box, color=(0, 0, 0))

    for i, box in enumerate(y_pred_):
        image = rectangle(image, box, color=(255, 0, 0))

    # Save the image
    cv2.imwrite(odir, image)


def draw_anchors(images, anchors, odir="anchors"):
    os.makedirs(odir, exist_ok=True)
    for i, (image, anchors) in enumerate(zip(images, anchors)):
        draw_anchors_on_image(image, anchors, f"{odir}/image-{i}-anchors.jpg")


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
        for name, module in self.sublosses.items():
            self.register_module(name, module.loss.keywords["loss_function"])
        self.register_buffer("anchors", anchors[None])
        self.count = 0

    def forward(self, predictions, targets):
        self.count += 1
        y = {
            "classes": stack([target["labels"] for target in targets]).long(),
            "boxes": stack([target["boxes"] for target in targets]),
            "keypoints": stack(
                [target["keypoints"] for target in targets],
            ),
            "depths": stack([target["depths"] for target in targets]),
        }
        # images = targets[0]["images"]
        y_pred = {
            "classes": predictions[1],
            "boxes": predictions[0],
            "keypoints": predictions[2],
            "depths": predictions[3],
        }

        positives, negatives = match(
            y["boxes"],
            (y["classes"] < 0).squeeze(-1),
            point_form(self.anchors),
        )

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

            y_pred_, y_true_, anchor_, n_pos_ = select(
                y_pred[name],
                y[name],
                self.anchors,
                use_negatives=subloss.needs_negatives,
                positives=positives,
                negatives=negatives,
            )
            # if name == "boxes":
            #     print(y_true_)
            #     draw_anchors_on_image(
            #         images[0],
            #         point_form(anchor_),
            #         y_true_,
            #         decode(y_pred_, anchor_, [0.1, 0.2]),
            #         f"debug-{self.count}.jpg",
            #     )
            # Plot the images and the selected anchors, here
            losses[name] = subloss(y_pred_, y_true_, anchor_) / max(n_pos_, 1)
            if not torch.isfinite(losses[name]).all():
                print(name, losses[name], y_pred_, y_true_)

        losses["loss"] = torch.stack(tuple(losses.values())).sum()
        return losses
