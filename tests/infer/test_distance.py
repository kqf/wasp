import numpy as np
import pytest

from wasp.infer.distance import distance2bbox, distance2kps


@pytest.fixture
def sample(max_shape):
    points = np.array([[50, 50], [100, 100], [200, 150]])
    distance = np.array(
        [
            [-10, -10, 20, 20],
            [0, 0, 30, 30],
            [-5, -15, 25, 10],
        ]
    )
    return points, distance, max_shape


@pytest.mark.parametrize(
    "max_shape, expected",
    [
        (
            None,
            [
                [60, 60, 70, 70],
                [100, 100, 130, 130],
                [205, 165, 225, 160],
            ],
        ),
        (
            (200, 200),
            [
                [60, 60, 70, 70],
                [100, 100, 130, 130],
                [200, 165, 200, 160],
            ],
        ),
    ],
)
def test_distance2bbox(sample, expected):
    decoded = distance2bbox(*sample)
    assert np.allclose(decoded, np.array(expected), atol=1e-6)


@pytest.mark.skip
@pytest.mark.parametrize("max_shape", [None, (200, 200)])
def test_distance2kps(sample):
    decoded = distance2kps(*sample)
    expected = np.array(
        [
            [40, 40, 70, 70],
            [100, 100, 130, 130],
            [195, 135, 200, 160],
        ]
    )
    assert np.allclose(decoded, expected, atol=1e-6)
