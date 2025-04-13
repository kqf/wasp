import torch
import torch.nn.functional as F

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


def match_boxes(
    boxes: torch.Tensor,  # [n_obj, 4]
    priors: torch.Tensor,  # [n_anchors, 4]
    overlap_threshold: float,
) -> torch.Tensor:  # returns a tensor of shape [n_anchors, n_obj]
    n_anchors = priors.shape[0]
    n_obj = boxes.shape[0]

    # Compute IoU overlaps: [n_obj, n_anchors]
    overlaps = iou(boxes[:, None], point_form(priors))

    # For each ground truth box, find the best matching prior (anchor)
    best_prior_overlap, best_prior_idx = overlaps.max(dim=1, keepdim=True)
    valid_gt_idx = best_prior_overlap[:, 0] >= 0.2

    # For each prior (anchor), find the best matching ground truth box
    best_truth_overlap, best_truth_idx = overlaps.max(dim=0, keepdim=True)
    best_truth_overlap = best_truth_overlap.squeeze(0)
    best_truth_idx = best_truth_idx.squeeze(0)
    best_prior_idx = best_prior_idx.squeeze(1)

    # Even if no good matches, return a zero-filled match matrix
    matching_table = torch.zeros(
        (n_anchors, n_obj), dtype=torch.bool, device=boxes.device
    )
    if valid_gt_idx.sum() == 0:
        return matching_table

    # Force match: assign each valid GT to its best matching prior
    best_truth_overlap.index_fill_(0, best_prior_idx[valid_gt_idx], 2)
    best_truth_idx[best_prior_idx] = torch.arange(
        best_prior_idx.shape[0], device=best_prior_idx.device
    )

    # Mark anchors as positive if IoU exceeds threshold
    valid_anchors = best_truth_overlap >= overlap_threshold
    matching_table[valid_anchors, best_truth_idx[valid_anchors]] = 1
    return matching_table


def mine_negatives(
    label: torch.Tensor,
    pred: torch.Tensor,
    negpos_ratio: int,
    positive: torch.Tensor,
) -> torch.Tensor:
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


def match(
    classes: torch.Tensor,  # [batch_size, n_anchors, n_classes]
    boxes: torch.Tensor,  # [batch_size, n_obj, 4]
    priors: torch.Tensor,  # [n_anchors, 4]
    confidences: torch.Tensor,  # [batch_size, n_anchors, n_classes]
    negpos_ratio: int,
    overalp: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    positives = torch.stack([match_boxes(b, priors, overalp) for b in boxes])
    negatives = mine_negatives(classes, confidences, negpos_ratio, positives)
    return positives, negatives
