import os

import cv2
import numpy as np
import pytest

from wasp.infer.detection.arcface import ArcFaceONNX, Face

MODEL = "models/w600k_r50.onnx"


def model_exists():
    return os.path.exists(MODEL)


@pytest.fixture
def image() -> np.ndarray:
    return cv2.cvtColor(
        cv2.imread("tests/assets/lenna.png"),
        cv2.COLOR_BGR2RGB,
    )


@pytest.fixture
def model() -> ArcFaceONNX:
    return ArcFaceONNX(MODEL)


@pytest.mark.skipif(
    not model_exists(),
    reason="File doesn't exists, skipping the test",
)
def test_srfd_inferencd(model: ArcFaceONNX, image: np.ndarray):
    face = Face(
        kps=np.array(
            [
                [269.48657, 267.8749],
                [334.58054, 266.53412],
                [313.75803, 321.2535],
                [271.43204, 349.04657],
                [315.51984, 347.4897],
            ]
        )
    )
    representation = model.get(image, face)
    assert representation.shape == 1
