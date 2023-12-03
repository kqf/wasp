import pytest
import torch

from wasp.retinaface.loss import MultiBoxLoss
from wasp.retinaface.priors import priorbox


@pytest.fixture
def anchors(resolution=(256, 256)):
    return priorbox(
        min_sizes=[[16, 32], [64, 128], [256, 512]],
        steps=[8, 16, 32],
        clip=False,
        image_size=resolution,
    )


@pytest.fixture
def loss(anchors):
    return MultiBoxLoss(
        num_classes=2,
        overlap_thresh=0.35,
        prior_for_matching=True,
        bkg_label=0,
        neg_mining=True,
        neg_pos=7,
        neg_overlap=0.35,
        encode_target=False,
        priors=anchors,
        weights={
            "localization": 2,
            "classification": 1,
            "landmarks": 1,
        },
    )


@pytest.fixture
def predictions():
    x = [
        torch.zeros([1, 2688, 4]),
        torch.zeros([1, 2688, 2]),
        torch.zeros([1, 2688, 10]),
    ]
    x[0][0, 0] = 1.0
    return x


@pytest.fixture
def targets():
    return [torch.zeros((1, 15))]


def test_loss(loss, predictions, targets):
    losses = loss.full_forward(predictions, targets)
    assert len(losses) == 4
