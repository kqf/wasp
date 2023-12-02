import pytest

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


def test_loss(loss):
    pass
