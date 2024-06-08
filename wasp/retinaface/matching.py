import torch

from wasp.retinaface.encode import point_form


def intersect(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """Computes the area of intersection between each pair of boxes."""
    a, b = box_a.size(0), box_b.size(0)
    max_xy = torch.min(
        box_a[:, 2:].unsqueeze(1).expand(-1, b, -1),
        box_b[:, 2:].unsqueeze(0).expand(a, -1, -1),
    )
    min_xy = torch.max(
        box_a[:, :2].unsqueeze(1).expand(-1, b, -1),
        box_b[:, :2].unsqueeze(0).expand(a, -1, -1),
    )
    inter = torch.clamp(max_xy - min_xy, min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def iou(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """Computes the Intersection over Union (IoU) of two sets of boxes."""
    inter = intersect(box_a, box_b)
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
    union = area_a.unsqueeze(1) + area_b.unsqueeze(0) - inter
    return inter / union


def match(
    labels: torch.Tensor,
    boxes: torch.Tensor,
    priors: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    overlaps = iou(boxes, point_form(priors))

    # (Bipartite Matching)
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    valid_gt_idx = best_prior_overlap[:, 0] >= 0.2

    if valid_gt_idx.sum() == 0:
        return None, None

    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_overlap = best_truth_overlap.squeeze(0)
    best_prior_idx = best_prior_idx.squeeze(1)
    best_truth_overlap.index_fill_(0, best_prior_idx[valid_gt_idx], 2)

    best_truth_idx = best_truth_idx.squeeze(0)

    # Use arange instead of for loop
    best_truth_idx[best_prior_idx] = torch.arange(
        best_prior_idx.shape[0],
        device=best_prior_idx.device,
    )

    labels_matched = labels[best_truth_idx].clone()
    labels_matched[best_truth_overlap < threshold] = 0

    return best_truth_idx, labels_matched
