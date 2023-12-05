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
    x[1][0, 0, 0] = 1.0
    return x


@pytest.fixture
def targets():
    x = torch.zeros((1, 15))
    x[0, :4] = torch.Tensor([0.0020, 0.6445, 0.1230, 0.9980])
    x[0, -1] = 1.0
    return [x]


def test_loss(loss, predictions, targets):
    total, boxes, classes, landmarks = loss.full_forward(predictions, targets)
    assert total == 0
    assert boxes == 0
    assert classes == 0
    assert landmarks == 0
