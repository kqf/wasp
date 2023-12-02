import pytest

from wasp.retinaface.loss import MultiBoxLoss


@pytest.fixture
def loss(priors):
    return MultiBoxLoss(
        num_classes=2,
        overlap_thresh=0.35,
        prior_for_matching=True,
        bkg_label=0,
        neg_mining=True,
        neg_pos=7,
        neg_overlap=0.35,
        encode_target=False,
        priors=priors,
        weights={
            "localization": 2,
            "classification": 1,
            "landmarks": 1,
        },
    )


def test_loss(loss):
    pass
