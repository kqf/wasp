import torch

from wasp.retinaface.priors import priorbox

RESOLUTION = 480, 640


def test_anchors(resolution=RESOLUTION):
    """Check all anchors"""
    anchors = priorbox(
        min_sizes=[[16, 32], [64, 128], [256, 512]],
        steps=[8, 16, 32],
        clip=False,
        image_size=resolution,
    )
    print(anchors.shape)
    shapes = anchors[:, 2:] * torch.Tensor([resolution])
    shapes = list(map(tuple, shapes.tolist()))
    print(set(shapes))


def test_anchors_simplified(resolution=RESOLUTION):
    """Check no small anchors"""
    anchors = priorbox(
        min_sizes=[[64, 128], [256, 512]],
        steps=[16, 32],
        clip=False,
        image_size=resolution,
    )
    print(anchors.shape)
    shapes = anchors[:, 2:] * torch.Tensor([resolution])
    shapes = list(map(tuple, shapes.tolist()))
    print(set(shapes))
