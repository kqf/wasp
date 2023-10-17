from typing import List

import torch


def iou(*args):
    pass


def point_form(*args):
    pass


def encode(*args):
    pass


def encode_landm(*args):
    pass


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
