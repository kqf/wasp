import pytest
import torch

from wasp.retinaface.model import RetinaFace


@pytest.mark.parametrize(
    "inputs",
    [
        torch.randn(1, 3, 640, 480),  # Random input image
        torch.randn(1, 3, 1280, 720),  # Random input image
    ],
)
def test_retinaface_forward(inputs):
    model = RetinaFace(
        name="Resnet50",
        pretrained=False,
        in_channels=256,
        return_layers={"layer1": 0, "layer2": 1, "layer3": 2},
        out_channels=256,
    )

    # Act
    bbox_regressions, classifications, ldm_regressions = model.forward(inputs)

    # Assert
    assert bbox_regressions.shape == (
        inputs.shape[0],
        12,
        inputs.shape[2] // 4,
        inputs.shape[3] // 4,
    )
    assert classifications.shape == (
        inputs.shape[0],
        6,
        inputs.shape[2] // 4,
        inputs.shape[3] // 4,
    )
    assert ldm_regressions.shape == (
        inputs.shape[0],
        10,
        inputs.shape[2] // 4,
        inputs.shape[3] // 4,
    )
