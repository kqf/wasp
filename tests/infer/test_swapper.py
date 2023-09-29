import os

import numpy as np
import pytest

from wasp.infer.swapper import INSwapper

target = np.zeros((512,))
source = np.ones((512,))


@pytest.fixture
def swapper() -> INSwapper:
    return INSwapper(MODEL, None)


MODEL = "models/inswapper_128.onnx"


def model_exists():
    return os.path.exists(MODEL)


@pytest.mark.skipif(
    not model_exists(),
    reason="File doesn't exists, skipping the test",
)
def test_get(swapper, image, target, source):
    result = swapper.get(image, target, source, paste_back=True)

    # Assert
    assert isinstance(result, tuple)
    assert isinstance(result[0], np.ndarray)
    assert isinstance(result[1], np.ndarray)
