import pytest
import torch
import torchvision

from wasp.retinaface.model import RetinaFace


@pytest.mark.parametrize(
    "inputs, anchors",
    [
        (torch.randn(1, 3, 640, 480), 12600),
        (torch.randn(1, 3, 1280, 720), 37840),
    ],
)
def test_retinaface(inputs, anchors):
    model = RetinaFace(
        name="Resnet50",
        pretrained=False,
        return_layers={"layer2": 1, "layer3": 2, "layer4": 3},
        in_channels=256,
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
        torch.randn(1, 3, 1280, 720),
    ],
)
def test_backbone(image):
    model = torchvision.models.resnet50(weights=None)
    output = model(image)
    print(output.shape)
