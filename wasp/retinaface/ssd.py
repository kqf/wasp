from functools import partial
from typing import Any, Optional

import numpy as np
import torch
import torchvision
import tqdm
from environs import Env
from torch.utils.data import DataLoader
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.ssdlite import (
    SSD,
    MobileNet_V3_Large_Weights,
    SSDLiteHead,
    _mobilenet_extractor,
    mobilenet_v3_large,
)

import wasp.retinaface.augmentations as augs
from wasp.retinaface.data import FaceDetectionDataset, detection_collate
from wasp.retinaface.preprocess import compose, normalize, preprocess


# Transformations adjusted for 640x640 images
def get_transform(train):
    transforms = [
        torchvision.transforms.Resize((640, 480)),
        torchvision.transforms.ToTensor(),
    ]
    if train:
        transforms.append(torchvision.transforms.RandomHorizontalFlip(0.5))
    return torchvision.transforms.Compose(transforms)


def convert(key, value, mask):
    if key == "labels":
        return value.squeeze(1).to(torch.int64)[mask]
    return value if key == "images" else value[mask]


def vis_outputs(images, boxes):
    import cv2

    from wasp.retinaface.data import Annotation
    from wasp.retinaface.visualize.plot import plot, to_image

    vis = plot(
        image=to_image(images[-1]),
        annotations=[Annotation(bbox, ()) for bbox in boxes],
    )
    cv2.imwrite("debug.jpg", vis)


class SSDModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return (x, None) if self.model.training else (x, self.model(x))

    def loss(self, batch_outs, targets):
        batch, _ = batch_outs
        converted = []
        for entry in targets:
            boxes = entry["boxes"]
            mask = (boxes[:, 2:] > boxes[:, :2]).all(-1)
            converted.append(
                {k: convert(k, v, mask) for k, v in entry.items()},
            )

        if self.model.training:
            losses: dict = self.model(batch, converted)
        else:
            self.model.train()
            with torch.no_grad():
                losses: dict = self.model(batch, converted)
            self.model.eval()
            # Don't do anything with losses
            # losses = {"classification": 0, "bbox_regression": 0}

        total = sum(losses.values())
        return {
            "loss": total,
            "classes": losses["classification"],
            "boxes": losses["bbox_regression"],
        }

    def prepare_outputs(self, images, out, targets, prior_box):
        _, out = out
        total = []
        image_height = images.shape[2]
        image_width = images.shape[3]
        scale = np.tile([image_width, image_height], 2)
        for preds, labels in zip(out, targets):
            boxes = preds["boxes"].cpu().numpy()
            scores = preds["scores"].cpu().numpy()
            candidates = np.concatenate(
                (boxes, scores.reshape(-1, 1), scores.reshape(-1, 1)),
                axis=1,
            )
            candidates[:, -2] = 0

            boxes_gt = labels["boxes"].cpu().numpy()
            labels_gt = labels["labels"].cpu().numpy()
            gts = np.zeros((boxes_gt.shape[0], 7), dtype=np.float32)
            gts[:, :4] = boxes_gt[:, :4] * scale[None, :]
            gts[:, 4] = np.where(labels_gt[:, -1] > 0, 0, 1)
            total.append((candidates, gts))

        # Uncomment to debug the plots
        # vis_outputs(images, boxes)
        return total


def load_with_mismatch(model, pretrained_state_dict):
    model_state_dict = model.state_dict()

    for name, pretrained_param in pretrained_state_dict.items():
        if name in model_state_dict:
            if "classification_head" in name:
                continue

            model_param = model_state_dict[name]
            if pretrained_param.shape != model_param.shape:
                ns = model_param.shape
                expanded = pretrained_param
                for dim in range(len(ns)):
                    if pretrained_param.shape[dim] != ns[dim]:
                        repeats = ns[dim] // pretrained_param.shape[dim]
                        expanded = expanded.repeat_interleave(repeats, dim=dim)
                pretrained_param = expanded
            model_state_dict[name] = pretrained_param.clone()

    model.load_state_dict(model_state_dict)
    return model


def ssdlite320_mobilenet_v3_large_custom(
    *,
    size: tuple[int, int],
    weights: Optional[SSDLite320_MobileNet_V3_Large_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    weights_backbone: Optional[
        MobileNet_V3_Large_Weights
    ] = MobileNet_V3_Large_Weights.IMAGENET1K_V1,
) -> SSD:
    weights = SSDLite320_MobileNet_V3_Large_Weights.verify(weights)
    weights_backbone = MobileNet_V3_Large_Weights.verify(weights_backbone)
    # Enable reduced tail if no pretrained backbone is selected.
    # See Table 6 of MobileNetV3 paper.
    reduce_tail = weights_backbone is None

    norm_layer = partial(torch.nn.BatchNorm2d, eps=0.001, momentum=0.03)
    backbone = mobilenet_v3_large(
        weights=weights_backbone,
        progress=progress,
        norm_layer=norm_layer,
        reduced_tail=reduce_tail,
    )
    backbone = _mobilenet_extractor(
        backbone,
        6,
        norm_layer,
    )

    anchor_generator = DefaultBoxGenerator(
        [[2, 3] for _ in range(6)],
        min_ratio=0.2,
        max_ratio=0.95,
    )
    out_channels = det_utils.retrieve_out_channels(backbone, size)
    num_anchors = anchor_generator.num_anchors_per_location()

    defaults = {
        "score_thresh": 0.5,
        "nms_thresh": 0.4,
        "detections_per_img": 300,
        "topk_candidates": 300,
        # Rescale the input in a way compatible to the backbone:
        # The following mean/std rescale the data from [0, 1] to [-1, 1]
        "image_mean": [0.0, 0.0, 0.0],
        "image_std": [1.0, 1.0, 1.0],
        "_skip_resize": True,
    }
    kwargs: Any = {**defaults}
    model = SSD(
        backbone,
        anchor_generator,
        size,
        num_classes,
        head=SSDLiteHead(out_channels, num_anchors, num_classes, norm_layer),
        **kwargs,
    )
    return load_with_mismatch(model, weights.get_state_dict(progress=progress))


def build_model(
    resolution=(320, 320),
    num_classes=2,
) -> SSDModel:
    # Load pre-trained model
    model = ssdlite320_mobilenet_v3_large_custom(
        weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT,
        size=resolution,
        num_classes=num_classes,
    )
    return SSDModel(model)


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
    model = build_model(resolution)
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
