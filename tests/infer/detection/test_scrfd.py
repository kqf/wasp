import os

import cv2
import numpy as np
import pytest

from wasp.infer.detection.scrfd import SCRFD

MODEL = "models/det_10g.onnx"


def model_exists():
    return os.path.exists(MODEL)


@pytest.fixture
def image() -> np.ndarray:
    return cv2.imread("tests/assets/lenna.png")


@pytest.fixture
def model() -> SCRFD:
    model = SCRFD(MODEL)
    model.prepare(0, det_rhesh=0.5, input_size=(512, 512))
    return model


@pytest.mark.skipif(
    not model_exists(),
    reason="File doesn't exists, skipping the test",
)
def test_srfd_inferencd(model: SCRFD, image: np.ndarray):
    bboxes, kps = model.detect(image, max_num=0)
    print(bboxes)
    print(kps)
