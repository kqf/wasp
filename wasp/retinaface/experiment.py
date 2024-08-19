from functools import partial

import torch
from environs import Env
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

import wasp.retinaface.augmentations as augs
from wasp.retinaface.closs import DetectionLoss
from wasp.retinaface.data import FaceDetectionDataset, detection_collate

# from wasp.retinaface.loss import MultiBoxLoss
from wasp.retinaface.model import RetinaFace
from wasp.retinaface.preprocess import compose, normalize, preprocess
from wasp.retinaface.priors import priorbox


def main():
    env = Env()
    env.read_env()

    # Define the necessary components
    train_labels = env.str("TRAIN_LABEL_PATH")
    # valid_labels = env.str("VALID_LABEL_PATH")

    resolution = (768, 768)
    model = RetinaFace(
        name="Resnet50",
        pretrained=True,
        return_layers={
            "layer2": 1,
            "layer3": 2,
            "layer4": 3,
        },
        out_channels=256,
    )

    priors = priorbox(
        min_sizes=[[16, 32], [64, 128], [256, 512]],
        steps=[8, 16, 32],
        clip=False,
        image_size=resolution,
    )

    loss_fn = DetectionLoss(anchors=priors)
    preprocessing = partial(preprocess, img_dim=resolution[0])

    # Initialize optimizer and scheduler
    optimizer = partial(
        torch.optim.Adam,
        lr=0.001,
        weight_decay=0.0001,
    )(params=[x for x in model.parameters() if x.requires_grad])

    scheduler = partial(
        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        T_0=10,
        T_mult=2,
    )(optimizer=optimizer)

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

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model.to(device)
    loss_fn.to(device)

    # Initialize gradient scaler for AMP
    scaler = GradScaler()

    # Training loop
    num_epochs = 10

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            targets = [
                {key: value.to(device) for key, value in entry.items()}
                for entry in batch["annotation"]
            ]

            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                losses = loss_fn(outputs, targets)
                loss = losses["loss"]
                print(f"Loss computed: {loss.item()}")  # Debug print

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}]",
                    f"Step [{batch_idx + 1}/{len(train_loader)}]",
                    f"Loss: {loss.item():.4f}",
                )

        scheduler.step()
        print(
            f"Epoch [{epoch + 1}/{num_epochs}]",
            f"Average Loss: {running_loss / len(train_loader):.4f}",
        )

    print("Training complete.")


if __name__ == "__main__":
    main()
