import pytest
import torch

from wasp.retinaface.data import DetectionTargets
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
        priors=anchors,
    )


@pytest.fixture
def predictions():
    x = [
        torch.zeros([1, 2688, 4]),
        torch.zeros([1, 2688, 2]),
        torch.zeros([1, 2688, 10]),
        torch.zeros([1, 2688, 2]),
    ]
    x[0][0, 0] = 1.0
    x[1][0, 0, 0] = 1.0
    return x


@pytest.fixture
def targets():
    x = torch.zeros((1, 17))
    x[0, :4] = torch.Tensor([0.0020, 0.6445, 0.1230, 0.9980])
    x[0, -2] = 1.0
    return DetectionTargets(boxes=x, classes=x, landmarks=x, depths=x)


@pytest.mark.skip
def test_loss(loss, predictions, targets):
    total, boxes, classes, landmarks, depths = loss.forward(
        predictions, targets
    )
    assert total == 0
    assert boxes == 0
    assert classes == 0
    assert landmarks == 0
