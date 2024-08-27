from functools import partial

import numpy as np
import torch
import torchvision
from environs import Env
from torch.utils.data import DataLoader
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.ssdlite import SSDLiteHead
from torchvision.ops import nms

import wasp.retinaface.augmentations as augs
from wasp.retinaface.data import FaceDetectionDataset, detection_collate
from wasp.retinaface.preprocess import compose, normalize, preprocess


# Transformations adjusted for 640x640 images
def get_transform(train):
    transforms = []
    transforms.append(torchvision.transforms.Resize((640, 480)))
    transforms.append(torchvision.transforms.ToTensor())
    if train:
        transforms.append(torchvision.transforms.RandomHorizontalFlip(0.5))
    return torchvision.transforms.Compose(transforms)


def convert(key, value):
    if key == "labels":
        return value.squeeze(1).to(torch.int64)
    return value


class SSDModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return x

    def loss(self, batch, targets):
        converted = []
        for entry in targets:
            boxes = entry["boxes"]
            v = (boxes[:, 2:] > (boxes[:, :2] + 0.01)).any(-1)
            converted.append(
                {key: convert(key, value)[v] for key, value in entry.items()}
            )

        losses: dict = self.model(batch, converted)
        total = sum(loss for loss in losses.values())
        return {
            "loss": total,
            "classes": losses["classification"],
            "boxes": losses["bbox_regression"],
        }


def prepare_outputs(
    images,
    out,
    targets,
    prior_box,
) -> list[tuple[np.ndarray, np.ndarray]]:
    total = []
    for batch_id, target in enumerate(targets):
        preds = out[batch_id]
        boxes = preds["boxes"]
        scores = preds["scores"]
        # labels = preds["labels"]

        # do NMS
        keep = nms(boxes, scores, 0.4)
        boxes = boxes[keep, :].cpu().numpy()
        scores = scores[keep].cpu().numpy()
        candidates = np.concatenate(
            (boxes, scores.reshape(-1, 1), scores.reshape(-1, 1)),
            axis=1,
        )
        candidates[:, -2] = 0

        boxes_gt = target["boxes"].cpu().numpy()
        labels_gt = target["labels"].cpu().numpy()
        gts = np.zeros((boxes_gt.shape[0], 7), dtype=np.float32)
        gts[:, :4] = boxes_gt[:, :4]  # * scale[None, :].cpu().numpy()
        gts[:, 4] = np.where(labels_gt[:, -1] > 0, 0, 1)
        total.append((candidates, gts))

    return total


def build_model(
    resolution=(320, 320),
    num_classes=2,
) -> SSDModel:
    # Load pre-trained model
    model = ssdlite320_mobilenet_v3_large(
        weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT,
    )
    anchor_generator = DefaultBoxGenerator(
        [[2, 3] for _ in range(6)], min_ratio=0.2, max_ratio=0.95
    )
    size = (320, 320)
    out_channels = det_utils.retrieve_out_channels(model.backbone, size)
    num_anchors = anchor_generator.num_anchors_per_location()
    if len(out_channels) != len(anchor_generator.aspect_ratios):
        raise ValueError("Wrong aspect ratio")

    model.head = SSDLiteHead(
        out_channels,
        num_anchors,
        num_classes,
        partial(torch.nn.BatchNorm2d, eps=0.001, momentum=0.03),
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=6,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        collate_fn=detection_collate,
    )
    return train_loader


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

    def convert(key, value):
        if key == "labels":
            return value.to(device).squeeze(1).to(torch.int64)
        return value.to(device)

    for epoch in range(num_epochs):
        model.train()
        for batch in data_loader:
            images = batch["image"].to(device)
            targets = [
                {key: convert(key, value) for key, value in entry.items()}
                for entry in batch["annotation"]
            ]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

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
