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
        dim=-1,
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
    # dist b/t match center and prior's center

    # Reshape matched to [num_priors, 5, 2]
    matched = torch.reshape(matched, (matched.shape[0], 5, 2))

    # Expand priors to match the shape of matched using broadcasting
    # Change shape from [num_priors, 4] to [num_priors, 1, 4]

    priors = priors[:, None]

    # Calculate the distance between the match center and the prior's center
    g_cxcy = matched[:, :, :2] - priors[:, :, :2]

    # Encode variance
    g_cxcy /= variances[0] * priors[:, :, 2:]

    # Return target for smooth_l1_loss
    return g_cxcy.reshape(-1, 10)


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
