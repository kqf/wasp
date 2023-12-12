from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torchvision import models
from torchvision.models import _utils

from wasp.retinaface.fpn import FPN, SSH


class ClassHead(nn.Module):
    def __init__(self, in_channels: int = 512, num_anchors: int = 3) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(
            in_channels,
            num_anchors * 2,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, in_channels: int = 512, num_anchors: int = 3):
        super().__init__()
        self.conv1x1 = nn.Conv2d(
            in_channels,
            num_anchors * 4,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, in_channels: int = 512, num_anchors: int = 3):
        super().__init__()
        self.conv1x1 = nn.Conv2d(
            in_channels,
            num_anchors * 10,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 10)


def _make_classes(
    fpn_num: int = 3, in_channels: int = 64, anchor_num: int = 2
) -> nn.ModuleList:
    classhead = nn.ModuleList()
    for _ in range(fpn_num):
        classhead.append(ClassHead(in_channels, anchor_num))
    return classhead


def _make_bboxes(
    fpn_num: int = 3, in_channels: int = 64, anchor_num: int = 2
) -> nn.ModuleList:
    bboxhead = nn.ModuleList()
    for _ in range(fpn_num):
        bboxhead.append(BboxHead(in_channels, anchor_num))
    return bboxhead


def _make_landmarks(
    fpn_num: int = 3,
    in_channels: int = 64,
    anchor_num: int = 2,
) -> nn.ModuleList:
    landmarkhead = nn.ModuleList()
    for _ in range(fpn_num):
        landmarkhead.append(LandmarkHead(in_channels, anchor_num))
    return landmarkhead


class RetinaFace(nn.Module):
    def __init__(
        self,
        name: str,
        pretrained: bool,
        return_layers: Dict[str, int],
        out_channels: int,
        in_channels: Optional[list[int]] = None,
    ) -> None:
        super().__init__()

        if name.lower() == "resnet50":
            backbone = models.resnet50(pretrained=pretrained)
        elif name.lower() == "resnet18":
            backbone = models.resnet18(weights=None)
        elif name.lower() == "mobilenet_v2":
            backbone = models.mobilenet_v2(weights=None).features
        elif name.lower() == "mobilenet_v3_small":
            backbone = models.mobilenet_v3_small(weights=None).features
        else:
            raise NotImplementedError(
                f"Only Resnet50 backbone is supported but got {name}"
            )

        self.body = _utils.IntermediateLayerGetter(backbone, return_layers)

        def build_channels(in_channels=256) -> list[int]:
            in_channels_stage2 = in_channels
            return [
                in_channels_stage2 * 2,
                in_channels_stage2 * 4,
                in_channels_stage2 * 8,
            ]

        in_channels_list = in_channels or build_channels()
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.classes = _make_classes(fpn_num=3, in_channels=out_channels)
        self.boxes = _make_bboxes(fpn_num=3, in_channels=out_channels)
        self.keypoints = _make_landmarks(fpn_num=3, in_channels=out_channels)

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [
            feature1,
            feature2,
            feature3,
        ]
        bbox_regressions = torch.cat(
            [self.boxes[i](feature) for i, feature in enumerate(features)],
            dim=1,
        )
        classifications = torch.cat(
            [self.classes[i](feature) for i, feature in enumerate(features)],
            dim=1,
        )
        ldm_regressions = torch.cat(
            [
                self.keypoints[i](feature)
                for i, feature in enumerate(features)
            ],  # noqa
            dim=1,
        )

        # bbox_regressions = self.boxes[0](feature3)
        # classifications = self.classes[0](feature3)
        # ldm_regressions = self.keypoints[0](feature3)
        return bbox_regressions, classifications, ldm_regressions
