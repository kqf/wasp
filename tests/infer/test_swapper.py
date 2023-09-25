import os

import numpy as np
import pytest

from wasp.infer.swapper import INSwapper

target = np.zeros((512,))
source = np.ones((512,))


@pytest.fixture
def swapper():
    model_file = "path/to/model.onnx"
    session = None
    return INSwapper(model_file, session)


MODEL = "models/inswapper_128.onnx"


def model_exists():
    return os.path.exists(MODEL)


@pytest.mark.skipif(
    not model_exists(),
    reason="File doesn't exists, skipping the test",
)
@pytest.mark.parametrize(
    "img, target, source, paste_back",
    [
        # Test case 1
        (
            np.zeros((100, 100, 3), dtype=np.uint8),
            target,
            source,
            False,
        ),
        # Test case 2
        (
            np.ones((200, 200, 3), dtype=np.uint8),
            target,
            source,
            True,
        ),
        # Add more happy path test cases here
    ],
    ids=["test_case1", "test_case2"],
)
def test_get(swapper, img, target, source, paste_back):
    # Arrange

    # Act
    result = swapper.get(img, target, source, paste_back)

    # Assert
    assert isinstance(result, tuple)
    assert isinstance(result[0], np.ndarray)
    assert isinstance(result[1], np.ndarray)


# Edge cases


@pytest.mark.skipif(
    not model_exists(),
    reason="File doesn't exists, skipping the test",
)
@pytest.mark.parametrize(
    "img, target, source, paste_back",
    [
        # Test case 1
        (
            np.zeros((1, 1, 3), dtype=np.uint8),
            target,
            source,
            False,
        ),
        # Test case 2
        (
            np.ones((1000, 1000, 3), dtype=np.uint8),
            target,
            source,
            True,
        ),
        # Add more edge cases here
    ],
    ids=["test_case1", "test_case2"],
)
def test_get_edge_cases(swapper, img, target, source, paste_back):
    # Arrange

    # Act
    result = swapper.get(img, target, source, paste_back)

    # Assert
    assert isinstance(result, tuple)
    assert isinstance(result[0], np.ndarray)
    assert isinstance(result[1], np.ndarray)


# Error cases


@pytest.mark.skipif(
    not model_exists(),
    reason="File doesn't exists, skipping the test",
)
@pytest.mark.parametrize(
    "img, target, source, paste_back",
    [
        # Test case 1
        (
            None,
            target,
            source,
            False,
        ),
        # Test case 2
        (
            np.zeros((100, 100, 3), dtype=np.uint8),
            None,
            source,
            True,
        ),
        # Add more error cases here
    ],
    ids=["test_case1", "test_case2"],
)
def test_get_error_cases(swapper, img, target, source, paste_back):
    # Arrange

    # Act and Assert
    with pytest.raises(Exception):
        swapper.get(img, target, source, paste_back)
