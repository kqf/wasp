import pytest
import torch

from wasp.retinaface.loss import mine_negatives


# Simplified function using cross_entropy
def mine_negatives_cross_entropy(
    label,
    pred,
    negpos_ratio,
    num_classes,
    positive,
):
    batch_conf = pred.view(-1, num_classes)
    labels = label.view(-1)

    # Compute the classification loss using cross_entropy
    loss_c = torch.nn.functional.cross_entropy(
        batch_conf,
        labels,
        reduction="none",
    )
    n_batch = label.shape[0]

    # Hard Negative Mining
    loss_c[positive.view(-1)] = 0  # filter out positive boxes for now
    loss_c = loss_c.view(n_batch, -1)
    _, loss_idx = loss_c.sort(1, descending=True)
    _, idx_rank = loss_idx.sort(1)
    num_pos = positive.long().sum(1, keepdim=True)
    num_neg = torch.clamp(negpos_ratio * num_pos, max=positive.shape[1] - 1)
    return idx_rank < num_neg.expand_as(idx_rank)


# Sample data generator
@pytest.fixture
def label(batch_size, num_classes, n_anchors):
    return torch.randint(0, num_classes, (batch_size, n_anchors))


@pytest.fixture
def pred(batch_size, num_classes, n_anchors):
    return torch.randn(batch_size, n_anchors, num_classes)


@pytest.fixture
def positive(batch_size, n_anchors):
    return torch.randint(0, 2, (batch_size, n_anchors)).bool()


@pytest.mark.parametrize(
    "batch_size, n_anchors, num_classes",
    [
        (32, 8732, 3),
        (12, 10_000, 3),
    ],
)
def test_mine_negatives_results(
    label,
    pred,
    positive,
    num_classes,
    negpos_ratio=7,
):

    negatives_mask_gather = mine_negatives(
        label, pred, negpos_ratio, num_classes, positive
    )
    negatives_mask_cross_entropy = mine_negatives_cross_entropy(
        label, pred, negpos_ratio, num_classes, positive
    )

    assert torch.equal(
        negatives_mask_gather, negatives_mask_cross_entropy
    ), "The results of both methods should be the same."


@pytest.mark.parametrize(
    "batch_size, n_anchors, num_classes",
    [
        (32, 8732, 3),
        (12, 10_000, 3),
    ],
)
@pytest.mark.parametrize(
    "mining_function", [mine_negatives, mine_negatives_cross_entropy]
)
def test_performance(
    label,
    pred,
    positive,
    num_classes,
    mining_function,
    benchmark,
    negpos_ratio=7,
):
    benchmark(
        mining_function,
        label,
        pred,
        negpos_ratio,
        num_classes,
        positive,
    )
