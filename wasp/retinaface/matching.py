from typing import List, Tuple, Union

import torch


def intersect(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """We resize both tensors to [A,B,2] without new malloc.

    [A, 2] -> [A, 1, 2] -> [A, B, 2]
    [B, 2] -> [1, B, 2] -> [A, B, 2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: bounding boxes, Shape: [A, 4].
      box_b: bounding boxes, Shape: [B, 4].
    Return:
      intersection area, Shape: [A, B].
    """
    a = box_a.size(0)
    b = box_b.size(0)
    max_xy = torch.min(
        box_a[:, 2:].unsqueeze(1).expand(a, b, 2),
        box_b[:, 2:].unsqueeze(0).expand(a, b, 2),
    )
    min_xy = torch.max(
        box_a[:, :2].unsqueeze(1).expand(a, b, 2),
        box_b[:, :2].unsqueeze(0).expand(a, b, 2),
    )
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def iou(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """Computes the jaccard overlap of two sets of boxes.

    The jaccard overlap is simply the intersection over union of two boxes.
    Here we operate on ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = (
        ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]))
        .unsqueeze(1)
        .expand_as(inter)
    )  # [A,B]
    area_b = (
        ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1]))
        .unsqueeze(0)
        .expand_as(inter)
    )  # [A,B]
    union = area_a + area_b - inter
    return inter / union


def point_form(boxes: torch.Tensor) -> torch.Tensor:
    """Convert prior_boxes to (x_min, y_min, x_max, y_max) representation.

    For comparison to point form ground truth data.

    Args:
        boxes: center-size default boxes from priorbox layers.
    Return:
        boxes: Converted x_min, y_min, x_max, y_max form of boxes.
    """
    return torch.cat(
        (boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2),
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
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    # encode variance
    g_cxcy /= variances[0] * priors[:, 2:]
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


def to_shape(x, matched) -> torch.Tensor:
    return x.unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)


def encode_landm(
    matched: torch.Tensor,
    priors: torch.Tensor,
    variances: Union[List[float], Tuple[float, float]],
) -> torch.Tensor:
    """Encodes the variances from the priorbox
    layers into the ground truth boxes we have matched.

    (based on jaccard overlap) with the prior boxes.
    Args:
        matched: Coords of ground truth for each prior in point-form
            Shape: [num_priors, 10].
        priors: Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: Variances of priorboxes
    Return:
        encoded landmarks, Shape: [num_priors, 10]
    """
    # dist b/t match center and prior's center
    matched = torch.reshape(matched, (matched.size(0), 5, 2))
    priors_cx = to_shape(priors[:, 0], matched)
    priors_cy = to_shape(priors[:, 1], matched)
    priors_w = to_shape(priors[:, 2], matched)
    priors_h = to_shape(priors[:, 3], matched)
    priors = torch.cat([priors_cx, priors_cy, priors_w, priors_h], dim=2)
    g_cxcy = matched[:, :, :2] - priors[:, :, :2]
    # encode variance
    g_cxcy /= variances[0] * priors[:, :, 2:]
    # return target for smooth_l1_loss
    return g_cxcy.reshape(g_cxcy.size(0), -1)


def match(
    threshold: float,
    box_gt: torch.Tensor,
    priors: torch.Tensor,
    variances: List[float],
    labels_gt: torch.Tensor,
    landmarks_gt: torch.Tensor,
    box_t: torch.Tensor,
    label_t: torch.Tensor,
    landmarks_t: torch.Tensor,
    batch_id: int,
) -> None:
    # Compute iou between gt and priors
    overlaps = iou(box_gt, point_form(priors))
    # (Bipartite Matching)
    # [1, num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)

    # ignore hard gt
    valid_gt_idx = best_prior_overlap[:, 0] >= 0.2
    best_prior_idx_filter = best_prior_idx[valid_gt_idx, :]
    if best_prior_idx_filter.shape[0] <= 0:
        box_t[batch_id] = 0
        label_t[batch_id] = 0
        return

    # [1, num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_idx_filter.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    # ensure best prior
    best_truth_overlap.index_fill_(0, best_prior_idx_filter, 2)

    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    matches = box_gt[best_truth_idx]  # Shape: [num_priors, 4]
    labels = labels_gt[best_truth_idx]  # Shape: [num_priors]
    # label as background   overlap<0.35
    labels[best_truth_overlap < threshold] = 0
    loc = encode(matches, priors, variances)

    matches_landm = landmarks_gt[best_truth_idx]
    landmarks_gt = encode_landm(matches_landm, priors, variances)
    box_t[batch_id] = loc  # [num_priors, 4] encoded offsets to learn
    label_t[batch_id] = labels  # [num_priors] top class label for each prior
    landmarks_t[batch_id] = landmarks_gt


def log_sum_exp(*args, **kwargs):
    pass
