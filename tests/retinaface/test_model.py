from typing import Literal

import pytest
import torch
import torchvision

from wasp.retinaface.model import RetinaFace
from wasp.retinaface.priors import priorbox
from wasp.retinaface.ssd import RetinaNetPure, SSDPure


def check_shapes(model, image):
    keys = list(zip(*model.named_children()))[0][:-2]
    intermediate = torchvision.models._utils.IntermediateLayerGetter(
        model,
        {k: k for k in keys},
    )
    return intermediate(image)


@pytest.mark.parametrize(
    "inputs, anchors",
    [
        # All layers
        (torch.randn(1, 3, 640, 480), 12600),
        # (torch.randn(1, 3, 1280, 720), 37840),
        # Only two layers
        # (torch.randn(1, 3, 640, 480), 600),
        # (torch.randn(1, 3, 1280, 720), 1840),
    ],
)
@pytest.mark.parametrize(
    "name, return_layers, in_channels",
    [
        # ("resnet50", {"layer2": 1, "layer3": 2, "layer4": 3}, None),
        ("resnet18", {"layer2": 1, "layer3": 2, "layer4": 3}, [128, 256, 512]),
        # ("mobilenet_v2", {"6": 1, "13": 2, "16": 3}, [32, 96, 160]),
        # ("mobilenet_v3_small", {"3": 1, "8": 2, "10": 3}, [24, 48, 96]),
    ],
)
def test_retinaface(
    inputs: torch.Tensor,
    anchors: Literal[12600],
    name: Literal["resnet18"],
    return_layers: dict[str, int],
    in_channels: list[int],
):
    model = RetinaFace(
        name=name,
        return_layers=return_layers,
        in_channels=in_channels,
        out_channels=256,
    )

    total = sum(p.numel() for p in model.parameters())
    print(f"Model name {name}, size: {total:_}")

    bboxes, classes, landmarks, depths = model(inputs)
    assert bboxes.shape == (inputs.shape[0], anchors, 4)
    assert classes.shape == (inputs.shape[0], anchors, 2)
    assert landmarks.shape == (inputs.shape[0], anchors, 10)
    assert depths.shape == (inputs.shape[0], anchors, 2)


@pytest.mark.parametrize(
    "image",
    [
        torch.randn(1, 3, 640, 480),
        # torch.randn(1, 3, 1280, 720),
    ],
)
def test_backbone(image: torch.Tensor):
    model = torchvision.models.resnet50(weights=None)
    output = model(image)
    print(output.shape)

    pyramid = torchvision.models._utils.IntermediateLayerGetter(
        model,
        {"layer4": 3},
    )

    poutput = pyramid(image)
    # sourcery skip: no-loop-in-tests
    for k, v in poutput.items():
        print(k, v.shape)
    # Resnet
    # 1 torch.Size([1, 512, 80, 60])
    # 2 torch.Size([1, 1024, 40, 30])
    # 3 torch.Size([1, 2048, 20, 15])

    # Mobilenet_v2
    # 0 torch.Size([1, 32, 320, 240])
    # 1 torch.Size([1, 16, 320, 240])
    # 2 torch.Size([1, 24, 160, 120])
    # 3 torch.Size([1, 24, 160, 120])
    # 4 torch.Size([1, 32, 80, 60])
    # 5 torch.Size([1, 32, 80, 60])
    # 6 torch.Size([1, 32, 80, 60]) <<<<
    # 7 torch.Size([1, 64, 40, 30])
    # 8 torch.Size([1, 64, 40, 30])
    # 9 torch.Size([1, 64, 40, 30])
    # 10 torch.Size([1, 64, 40, 30])
    # 11 torch.Size([1, 96, 40, 30])
    # 12 torch.Size([1, 96, 40, 30])
    # 13 torch.Size([1, 96, 40, 30]) <<<<
    # 14 torch.Size([1, 160, 20, 15])
    # 15 torch.Size([1, 160, 20, 15])
    # 16 torch.Size([1, 160, 20, 15]) <<<<

    # Mobilenet v3
    # 0 torch.Size([1, 16, 320, 240])
    # 1 torch.Size([1, 16, 160, 120])
    # 2 torch.Size([1, 24, 80, 60])
    # 3 torch.Size([1, 24, 80, 60])  <<<<<<<<
    # 4 torch.Size([1, 40, 40, 30])
    # 5 torch.Size([1, 40, 40, 30])
    # 6 torch.Size([1, 40, 40, 30])
    # 7 torch.Size([1, 48, 40, 30])
    # 8 torch.Size([1, 48, 40, 30])  <<<<<<<<
    # 9 torch.Size([1, 96, 20, 15])
    # 10 torch.Size([1, 96, 20, 15]) <<<<<<<<

    print("---------------------------------")
    model2 = torchvision.models.mobilenet_v3_small(weights=None)
    test_outputs = check_shapes(model2.features, image)
    for k, v in test_outputs.items():
        print(k, v.shape)


@pytest.mark.parametrize(
    "inputs, anchors",
    [
        # All layers
        (torch.randn(1, 3, 640, 640), 12600),
        # (torch.randn(1, 3, 1280, 720), 37840),
        # Only two layers
        # (torch.randn(1, 3, 640, 480), 600),
        # (torch.randn(1, 3, 1280, 720), 1840),
    ],
)
def test_ssd(inputs: torch.Tensor, anchors: Literal[12600]):
    resolution = inputs.shape[-2:]
    print(resolution)
    # ref = ssdlite320_mobilenet_v3_large(
    #     pretrained=False, num_classes=2
    # )  # Change num_classes as needed
    # total = sum(p.numel() for p in ref.parameters())
    # ref.eval()

    # print(f"Model name ssd, size: {total:_}")
    # ref(inputs)

    model = SSDPure(resolution=resolution, n_classes=2)
    model.eval()
    total = sum(p.numel() for p in model.parameters())
    print(f"Model name ssd, size: {total:_}")
    # n_anchors = build_priors(resolution).shape[0]
    n_anchors = priorbox(
        min_sizes=[[64, 128], [256, 512], [1024, 2048]],
        steps=[16, 32, 64],
        clip=False,
        image_size=resolution,
    ).shape[0]
    bboxes, classes, landmarks, depths = model(inputs)
    assert bboxes.shape == (inputs.shape[0], n_anchors, 4)
    assert classes.shape == (inputs.shape[0], n_anchors, 2)
    assert landmarks.shape == (inputs.shape[0], n_anchors, 10)
    assert depths.shape == (inputs.shape[0], n_anchors, 2)


@pytest.mark.parametrize(
    "resolution",
    [
        [640, 480],
    ],
)
def test_compare_model_sizes(resolution):
    model = RetinaNetPure(resolution=resolution, n_classes=2)
    total = sum(p.numel() for p in model.parameters())
    model.eval()
    print(f"Model name ssd, size: {total:_}")
