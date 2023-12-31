import os

import numpy as np
import pytest

from wasp.infer.arcface import ArcFace

MODEL = "models/w600k_r50.onnx"


def model_exists():
    return os.path.exists(MODEL)


@pytest.fixture
def model() -> ArcFace:
    return ArcFace(MODEL)


@pytest.mark.skipif(
    not model_exists(),
    reason="File doesn't exists, skipping the test",
)
def test_srfd_inferencd(model: ArcFace, image: np.ndarray):
    keypoints = np.array(
        [
            [269.48657, 267.8749],
            [334.58054, 266.53412],
            [313.75803, 321.2535],
            [271.43204, 349.04657],
            [315.51984, 347.4897],
        ]
    )
    representation = model(image, keypoints)
    assert representation.shape == (512,)
    np.testing.assert_almost_equal(np.sum(representation), 10.953, 3)
    np.testing.assert_almost_equal(np.mean(representation), 0.021, 3)
    np.testing.assert_almost_equal(np.std(representation), 0.926, 3)
