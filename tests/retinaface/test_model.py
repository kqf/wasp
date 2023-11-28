import pytest
import torch
import torchvision

from wasp.retinaface.model import RetinaFace


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
        (torch.randn(1, 3, 640, 480), 12600),
        (torch.randn(1, 3, 1280, 720), 37840),
    ],
)
@pytest.mark.parametrize(
    "name, return_layers, in_channels",
    [
        ("resnet50", {"layer2": 1, "layer3": 2, "layer4": 3}, 256),
        ("resnet18", {"layer2": 1, "layer3": 2, "layer4": 3}, 64),
        ("mobilenet_v2", {"6": 1, "13": 2, "16": 3}, 16),
    ],
)
def test_retinaface(inputs, anchors, name, return_layers, in_channels):
    model = RetinaFace(
        name=name,
        pretrained=False,
        return_layers=return_layers,
        in_channels=in_channels,
        out_channels=256,
    )

    bboxes, classes, landmarks = model(inputs)
    assert bboxes.shape == (inputs.shape[0], anchors, 4)
    assert classes.shape == (inputs.shape[0], anchors, 2)
    assert landmarks.shape == (inputs.shape[0], anchors, 10)


@pytest.mark.parametrize(
    "image",
    [
        torch.randn(1, 3, 640, 480),
        # torch.randn(1, 3, 1280, 720),
    ],
)
def test_backbone(image):
    model = torchvision.models.resnet50(weights=None)
    output = model(image)
    print(output.shape)

    pyramid = torchvision.models._utils.IntermediateLayerGetter(
        model,
        {"layer2": 1, "layer3": 2, "layer4": 3},
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

    # Mobilenet

    print("---------------------------------")
    model2 = torchvision.models.mobilenet_v2(weights=None)
    test_outputs = check_shapes(model2.features, image)
    for k, v in test_outputs.items():
        print(k, v.shape)
