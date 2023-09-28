import cv2
import numpy as np
import pytest


@pytest.fixture
def image() -> np.ndarray:
    return cv2.cvtColor(
        cv2.imread("tests/assets/lenna.png"),
        cv2.COLOR_BGR2RGB,
    )
