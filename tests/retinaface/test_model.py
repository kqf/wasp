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


@pytest.mark.skip
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
    # 1 torch.Size([1, 512, 80, 60])
    # 2 torch.Size([1, 1024, 40, 30])
    # 3 torch.Size([1, 2048, 20, 15])

    print("---------------------------------")
    model2 = torchvision.models.resnet18(weights=None)
    test_outputs = check_shapes(model2, image)
    for k, v in test_outputs.items():
        print(k, v.shape)
