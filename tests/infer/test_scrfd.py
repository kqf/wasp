import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytest

from wasp.infer.scrfd import SCRFD, nninput, resize_image

from wasp.infer.scrfd import anchors_centers  # isort:skip


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
def model() -> SCRFD:
    return SCRFD(MODEL)


@pytest.mark.skipif(
    not model_exists(),
    reason="File doesn't exists, skipping the test",
)
def test_srfd_inferencd(model: SCRFD, image: np.ndarray):
    scores, boxes, keypoints = model.detect(image, max_num=0)
    # visualize(image, boxes[0], keypoints[0])

    assert boxes.shape == (1, 1, 4)
    assert keypoints.shape == (1, 1, 5, 2)

    # np.testing.assert_allclose(scores[0], 0.6662737)
    np.testing.assert_allclose(
        boxes,
        np.array(
            [[[207.30132, 178.63065, 353.27258, 389.34088]]],
        ),
    )

    print((keypoints[0][0]).__repr__())
    np.testing.assert_allclose(
        keypoints[0][0],
        np.array(
            [
                [269.48657, 267.8749],
                [334.58054, 266.53412],
                [313.75803, 321.2535],
                [271.43204, 349.04657],
                [315.51984, 347.4897],
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
    # plt.show()


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
