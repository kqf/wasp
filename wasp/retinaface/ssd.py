from functools import partial

import torch
import tqdm
from environs import Env
from torch.utils.data import DataLoader
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.ssdlite import (
    MobileNet_V3_Large_Weights,
    SSDLiteClassificationHead,
    SSDLiteRegressionHead,
    _mobilenet_extractor,
    mobilenet_v3_large,
)

import wasp.retinaface.augmentations as augs
from wasp.retinaface.data import FaceDetectionDataset, detection_collate
from wasp.retinaface.preprocess import compose, normalize, preprocess


class SSDPureHead(torch.nn.Module):
    def __init__(self, out_channels, num_anchors, norm_layer, n_classes):
        super().__init__()
        self.classification_head = SSDLiteClassificationHead(
            in_channels=out_channels,
            num_anchors=num_anchors,
            norm_layer=norm_layer,
            num_classes=n_classes,
        )
        self.regression_head = SSDLiteRegressionHead(
            in_channels=out_channels,
            num_anchors=num_anchors,
            norm_layer=norm_layer,
        )
        self.landmarks_head = SSDLiteClassificationHead(
            in_channels=out_channels,
            num_anchors=num_anchors,
            norm_layer=norm_layer,
            num_classes=10,
        )

        self.distances_head = SSDLiteClassificationHead(
            in_channels=out_channels,
            num_anchors=num_anchors,
            norm_layer=norm_layer,
            num_classes=2,
        )

    def forward(self, features):
        classes = self.classification_head(features)
        boxes = self.regression_head(features)
        landmarks = self.landmarks_head(features)
        distances = self.distances_head(features)
        return boxes, classes, landmarks, distances


class RetinaNetPureHead(torch.nn.Module):
    def __init__(self, out_channels, num_anchors, norm_layer, n_classes):
        super().__init__()
        from torchvision.models.detection.retinanet import (
            RetinaNetClassificationHead,
            RetinaNetRegressionHead,
        )

        self.classification_head = RetinaNetClassificationHead(
            in_channels=out_channels,
            num_anchors=num_anchors,
            norm_layer=norm_layer,
            num_classes=n_classes,
        )
        self.regression_head = RetinaNetRegressionHead(
            in_channels=out_channels,
            num_anchors=num_anchors,
            norm_layer=norm_layer,
        )
        self.landmarks_head = RetinaNetClassificationHead(
            in_channels=out_channels,
            num_anchors=num_anchors,
            norm_layer=norm_layer,
            num_classes=10,
        )

        self.distances_head = RetinaNetClassificationHead(
            in_channels=out_channels,
            num_anchors=num_anchors,
            norm_layer=norm_layer,
            num_classes=2,
        )

    def forward(self, features):
        classes = self.classification_head(features)
        boxes = self.regression_head(features)
        landmarks = self.landmarks_head(features)
        distances = self.distances_head(features)
        return boxes, classes, landmarks, distances


class RetinaNetPure(torch.nn.Module):
    def __init__(self, resolution, n_classes):
        super().__init__()
        from torchvision.models.detection.retinanet import (
            RetinaNet_ResNet50_FPN_V2_Weights,
        )
        from torchvision.models.resnet import ResNet50_Weights, resnet50

        backbone = resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V1,
            progress=True,
            norm_layer=torch.nn.BatchNorm2d,
        )
        # skip P2 because it generates too many anchors
        self.backbone = _resnet_fpn_extractor(
            backbone,
            5,
            returned_layers=[2, 3, 4],
        )

        self.head = RetinaNetPureHead(
            self.backbone.out_channels,
            2,
            n_classes=n_classes,
            norm_layer=partial(torch.nn.GroupNorm, 32),
        )
        load_with_mismatch(
            self,
            RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1.get_state_dict(),
        )

    def forward(self, images):
        features = self.backbone(images)
        features = list(features.values())[:-1]
        return self.head(features)


class SSDPure(torch.nn.Module):
    def __init__(self, resolution, n_classes):
        super().__init__()
        self.n_classes = n_classes
        weights_backbone = MobileNet_V3_Large_Weights.IMAGENET1K_V1

        norm_layer = partial(torch.nn.BatchNorm2d, eps=0.001, momentum=0.03)
        backbone = mobilenet_v3_large(
            weights=weights_backbone,
            progress=True,
            norm_layer=norm_layer,
            reduced_tail=False,
        )
        self.backbone = _mobilenet_extractor(
            backbone,
            6,
            norm_layer,
        )
        out_channels = det_utils.retrieve_out_channels(
            self.backbone,
            resolution,
        )
        num_anchors = [2 for _ in out_channels]
        self.head = SSDPureHead(
            out_channels=out_channels,
            num_anchors=num_anchors,
            norm_layer=norm_layer,
            n_classes=n_classes,
        )
        load_with_mismatch(
            self,
            SSDLite320_MobileNet_V3_Large_Weights.COCO_V1.get_state_dict(),
        )

    def forward(self, images):
        features = self.backbone(images)
        features = list(features.values())[:-3]
        return self.head(features)


def load_with_mismatch(model, pretrained_state_dict):
    def repeat(pretrained_param, model_param):
        if pretrained_param.shape == model_param.shape:
            return pretrained_param

        ns = model_param.shape
        expanded = pretrained_param
        for dim in range(len(ns)):
            if pretrained_param.shape[dim] < ns[dim]:
                repeats = ns[dim] // pretrained_param.shape[dim]
                expanded = expanded.repeat_interleave(repeats, dim=dim)
            if pretrained_param.shape[dim] > ns[dim]:
                expanded = torch.narrow(pretrained_param, dim, 0, ns[dim])
        return expanded

    model_state_dict = model.state_dict()

    for name, pretrained_param in pretrained_state_dict.items():
        if name in model_state_dict:
            model_state_dict[name] = repeat(
                pretrained_param.clone(),
                model_state_dict[name],
            )

    model.load_state_dict(model_state_dict)
    return model


def build_dataloader(resolution):
    env = Env()
    env.read_env()
    train_labels = env.str("TRAIN_LABEL_PATH")
    preprocessing = partial(preprocess, img_dim=resolution[0])
    # Data loaders
    train_dataset = FaceDetectionDataset(
        label_path=train_labels,
        transform=augs.train(resolution),
        preproc=compose(normalize, preprocessing),
        rotate90=False,
    )

    return DataLoader(
        train_dataset,
        batch_size=12,
        num_workers=10,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        collate_fn=detection_collate,
    )


def main(
    num_epochs=10,
    resolution=(320, 320),
):
    model = SSDPure()
    data_loader = build_dataloader(resolution)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1,
    )

    # Training loop
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in tqdm.tqdm(range(num_epochs)):
        model.train()
        for batch in tqdm.tqdm(data_loader):
            images = batch["image"].to(device)
            targets = [
                {key: value.to(device) for key, value in entry.items()}
                for entry in batch["annotation"]
            ]

            _ = model(images)  # noqa
            loss_dict = model.loss(images, targets)

            losses = sum(loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        lr_scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {losses.item()}")

    # Save the fine-tuned model
    torch.save(
        model.state_dict(),
        "ssdlite640_mobilenet_v3_large_finetuned.pth",
    )


if __name__ == "__main__":
    main()
