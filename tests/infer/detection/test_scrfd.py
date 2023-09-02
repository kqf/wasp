import os

import cv2
import numpy as np
import pytest

MODEL = "models/det_10g.onnx"


def model_exists():
    return os.path.exists(MODEL)


@pytest.fixture
def image() -> np.ndarray:
    return cv2.imread("tests/assets/lenna.png")


@pytest.mark.skipif(
    not model_exists(),
    reason="File doesn't exists, skipping the test",
)
def test_srfd_inferencd(image: np.ndarray):
    print("HERE", image.shape)
