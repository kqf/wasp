import pytest
import torch

from wasp.retinaface.priors import priorbox

RESOLUTION = 480, 640


@pytest.mark.parametrize(
    "min_sizes, steps",
    [
        ([[16, 32], [64, 128], [256, 512]], [8, 16, 32]),
        ([[64, 128], [256, 512]], [16, 32]),
    ],
)
def test_anchors(min_sizes, steps, resolution=RESOLUTION):
    """Check all anchors"""
    anchors = priorbox(
        min_sizes=min_sizes,
        steps=steps,
        clip=False,
        image_size=resolution,
    )
    print(anchors.shape)
    shapes = anchors[:, 2:] * torch.Tensor([resolution])
    shapes = list(map(tuple, shapes.tolist()))
    print(set(shapes))
