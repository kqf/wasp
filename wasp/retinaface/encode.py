from typing import List, Tuple, Union

import torch


def point_form(boxes: torch.Tensor) -> torch.Tensor:
    """Convert prior_boxes to (x_min, y_min, x_max, y_max) representation.

    For comparison to point form ground truth data.

    Args:
        boxes: center-size default boxes from priorbox layers.
    Return:
        boxes: Converted x_min, y_min, x_max, y_max form of boxes.
    """
    return torch.cat(
        (
            boxes[..., :2] - boxes[..., 2:] / 2,
            boxes[..., :2] + boxes[..., 2:] / 2,
        ),
        dim=1,
    )


def encode(
    matched: torch.Tensor,
    priors: torch.Tensor,
    variances: List[float],
) -> torch.Tensor:
    """Encodes the variances from the priorbox layers into the gt boxes matched

     (based on jaccard overlap) with the prior boxes.
    Args:
        matched: Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: Variances of priorboxes
    Return:
        encoded boxes, Shape: [num_priors, 4]
    """
    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[..., :2]
    # encode variance
    g_cxcy /= variances[0] * priors[..., 2:]
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[..., 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(
    loc: torch.Tensor,
    priors: torch.Tensor,
    variances: Union[List[float], Tuple[float, float]],
) -> torch.Tensor:
    """Decodes locations from predictions using priors to undo the encoding
        we did for offset regression at train time.

    Args:
        loc: location predictions for loc layers,
            Shape: [num_priors, 4]
        priors: Prior boxes in center-offset form.
            Shape: [num_priors, 4].
        variances: Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes = torch.cat(
        (
            priors[..., :2] + loc[..., :2] * variances[0] * priors[..., 2:],
            priors[..., 2:] * torch.exp(loc[..., 2:] * variances[1]),
        ),
        dim=-1,
    )
    boxes[..., :2] -= boxes[..., 2:] / 2
    boxes[..., 2:] += boxes[..., :2]

    return boxes


def to_shape(x, matched) -> torch.Tensor:
    return x.unsqueeze(1).expand(matched.shape[0], 5).unsqueeze(2)


def encode_landm(
    matched: torch.Tensor,
    priors: torch.Tensor,
    variances: Union[List[float], Tuple[float, float]],
) -> torch.Tensor:
    # Check initial shapes
    print(f"Initial matched shape: {matched.shape}")
    print(f"Initial priors shape: {priors.shape}")

    if matched.numel() == 0 or priors.numel() == 0:
        raise ValueError("Input tensors must not be empty")

    # Reshape matched tensor
    matched = torch.reshape(matched, (matched.shape[0], 5, 2))
    print(f"Reshaped matched shape: {matched.shape}")

    # Expand priors to match the shape of matched
    priors_cx = to_shape(priors[:, 0], matched)
    priors_cy = to_shape(priors[:, 1], matched)
    priors_w = to_shape(priors[:, 2], matched)
    priors_h = to_shape(priors[:, 3], matched)
    priors = torch.cat([priors_cx, priors_cy, priors_w, priors_h], dim=2)
    print(f"Priors shape after expansion: {priors.shape}")

    # Calculate g_cxcy
    g_cxcy = matched[:, :, :2] - priors[:, :, :2]
    print(f"g_cxcy shape before encoding variance: {g_cxcy.shape}")

    # Encode variance
    g_cxcy /= variances[0] * priors[:, :, 2:]
    print(f"g_cxcy shape after encoding variance: {g_cxcy.shape}")

    # Return target for smooth_l1_loss
    final_shape = g_cxcy.shape[0], -1
    print(f"Final reshape target shape: {final_shape}")
    return g_cxcy.reshape(final_shape)


def decode_landm(
    pre: torch.Tensor,
    priors: torch.Tensor,
    variances: Union[List[float], Tuple[float, float]],
) -> torch.Tensor:
    """Decodes landmark locations from predictions using priors to decode
    Args:
        loc_landm: Landmark location predictions for landmark layers,
            Shape: [num_priors, 10]
        priors: Prior boxes in center-offset form.
            Shape: [num_priors, 4].
        variances: Variances of priorboxes
    Return:
        decoded landmark predictions
    """
    return torch.cat(
        (
            priors[..., :2] + pre[..., :2] * variances[0] * priors[..., 2:],
            priors[..., :2] + pre[..., 2:4] * variances[0] * priors[..., 2:],
            priors[..., :2] + pre[..., 4:6] * variances[0] * priors[..., 2:],
            priors[..., :2] + pre[..., 6:8] * variances[0] * priors[..., 2:],
            priors[..., :2] + pre[..., 8:10] * variances[0] * priors[..., 2:],
        ),
        dim=-1,
    )
