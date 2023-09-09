import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytest

from wasp.infer.detection.scrfd import SCRFD, nninput, resize_image

from wasp.infer.detection.scrfd import anchors_centers  # isort:skip


MODEL = "models/det_10g.onnx"


def visualize(image, boxes, keypoints):
    dimg = image.copy()
    for box, kps in zip(boxes, keypoints):
        box = box.astype(int)
        color = (0, 0, 255)
        cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
        if kps is not None:
            kps = kps.astype(int)
            for p in range(kps.shape[0]):
                color = (0, 0, 255)
                if p in [0, 3]:
                    color = (0, 255, 0)
                cv2.circle(dimg, (kps[p][0], kps[p][1]), 1, color, 2)
    plt.imshow(dimg)
    plt.show()


def model_exists():
    return os.path.exists(MODEL)


@pytest.fixture
def image() -> np.ndarray:
    return cv2.cvtColor(
        cv2.imread("tests/assets/lenna.png"),
        cv2.COLOR_BGR2RGB,
    )


@pytest.fixture
def model() -> SCRFD:
    return SCRFD(MODEL)


@pytest.mark.skipif(
    not model_exists(),
    reason="File doesn't exists, skipping the test",
)
def test_srfd_inferencd(model: SCRFD, image: np.ndarray):
    boxes, keypoints = model.detect(image, max_num=0)
    # visualize(image, boxes, keypoints)

    assert len(boxes) == 1
    assert len(keypoints) == 1

    np.allclose(
        boxes[0],
        np.array(
            [211.89163, 181.62668, 352.67505, 390.07153, 0.6662737],
        ),
    )
    np.allclose(
        keypoints[0],
        np.array(
            [
                [270.33743, 267.0965],
                [333.82535, 268.21793],
                [311.80945, 320.73257],
                [272.43158, 348.65176],
                [316.40335, 349.71494],
            ]
        ),
    )


@pytest.mark.parametrize(
    "inshape",
    [
        (200, 200),
        (512, 512),
        (1024, 1024),
    ],
)
def test_resize_image_correct_output(image, inshape):
    # Call the resize_image function
    result, scale = resize_image(image, inshape)

    # _, axs = plt.subplots(1, 2)
    # axs[0].imshow(image)
    # axs[1].imshow(result)
    # plt.show()

    # Check if the result has the correct shape
    assert result.shape == (inshape[1], inshape[0], 3)

    # Check if the scale is calculated correctly
    assert scale == inshape[1] / image.shape[0]


def test_blibifies(image: np.ndarray) -> None:
    h, w, c = image.shape
    blob = nninput(image)
    assert blob.shape == (1, c, h, w)
    _, axs = plt.subplots(1, 2)
    axs[0].imshow(image)
    axs[1].imshow(blob[0].transpose(1, 2, 0))
    plt.show()


@pytest.mark.parametrize(
    "height, width, stride, num_anchors, expected_shape",
    [
        (5, 4, 2, 1, (20, 2)),  # Test case 1: num_anchors = 1
        (6, 6, 3, 3, (108, 2)),  # Test case 2: num_anchors > 1
        (3, 3, 1, 2, (18, 2)),  # Test case 3: Specific values
    ],
)
def test_anchors_centers(height, width, stride, num_anchors, expected_shape):
    result = anchors_centers(height, width, stride, num_anchors)
    assert result.shape == expected_shape
