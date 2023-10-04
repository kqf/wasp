import os

import numpy as np
import pytest

from wasp.face import Face
from wasp.infer.swapper import INSwapper


@pytest.fixture
def swapper() -> INSwapper:
    return INSwapper(MODEL, None)


MODEL = "models/inswapper_128.onnx"


def model_exists():
    return os.path.exists(MODEL)


@pytest.fixture
def keypoints():
    return np.array(
        [
            [269.48657, 267.8749],
            [334.58054, 266.53412],
            [313.75803, 321.2535],
            [271.43204, 349.04657],
            [315.51984, 347.4897],
        ]
    )


@pytest.fixture
def bbox():
    return np.array(
        [207.30132, 178.63065, 353.27258, 389.34088],
    )


@pytest.fixture
def source(bbox, keypoints):
    return Face(
        bbox=bbox,
        kps=keypoints,
        detection_score=0.9,
        embedding=np.ones(512),
    )


@pytest.fixture
def target(bbox, keypoints):
    return Face(
        bbox=bbox,
        kps=keypoints,
        detection_score=0.9,
        embedding=np.ones(512),
    )


@pytest.mark.skipif(
    not model_exists(),
    reason="File doesn't exists, skipping the test",
)
def test_get(swapper, image, target, source):
    result = swapper.get(image, target, source)
    np.testing.assert_almost_equal(np.mean(result), 127.86726506551106)
    np.testing.assert_almost_equal(np.std(result), 58.55407466634696)
    assert result.shape == image.shape
