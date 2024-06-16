import torch

from wasp.retinaface.encode import point_form


def intersect(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    # [batch, n_obj, n_anchors, 2]
    max_xy = torch.min(box_a[..., 2:], box_b[..., 2:])
    min_xy = torch.max(box_a[..., :2], box_b[..., :2])

    inter = torch.clamp(max_xy - min_xy, min=0)
    # [batch, n_obj, n_anchors]
    return inter[..., 0] * inter[..., 1]


def iou(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    # [batch, n_obj, n_anchors]
    inter = intersect(box_a, box_b)

    # [batch, n_obj]
    area_a = (box_a[..., 2] - box_a[..., 0]) * (box_a[..., 3] - box_a[..., 1])
    # [batch, n_anchors]
    area_b = (box_b[..., 2] - box_b[..., 0]) * (box_b[..., 3] - box_b[..., 1])
    union = area_a + area_b - inter

    # [batch, n_obj, n_anchors]
    return inter / union


def match(
    labels: torch.Tensor,  # [n_obj]
    boxes: torch.Tensor,  # [n_obj, 4]
    priors: torch.Tensor,  # [n_anchors, 4]
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    overlaps = iou(boxes, point_form(priors))  # [n_obj, n_anchors]

    # (Bipartite Matching)
    best_prior_overlap, best_prior_idx = overlaps.max(
        1, keepdim=True
    )  # [n_obj, 1], [n_obj, 1]
    valid_gt_idx = best_prior_overlap[:, 0] >= 0.2  # [n_obj]

    if valid_gt_idx.sum() == 0:
        return None, None

    best_truth_overlap, best_truth_idx = overlaps.max(
        0, keepdim=True
    )  # [1, n_anchors], [1, n_anchors]
    best_truth_overlap = best_truth_overlap.squeeze(0)  # [n_anchors]
    best_prior_idx = best_prior_idx.squeeze(1)  # [n_obj]
    best_truth_overlap.index_fill_(0, best_prior_idx[valid_gt_idx], 2)

    best_truth_idx = best_truth_idx.squeeze(0)  # [n_anchors]

    # Use arange instead of for loop
    best_truth_idx[best_prior_idx] = torch.arange(
        best_prior_idx.shape[0],
        device=best_prior_idx.device,
    )

    labels_matched = labels[best_truth_idx].clone()  # [n_anchors]
    labels_matched[best_truth_overlap < threshold] = 0

    return best_truth_idx, labels_matched
