from functools import partial
from pathlib import Path

import pytorch_lightning as pl
import torch
from environs import Env

from wasp.retinaface.checkpoint import BestModelCheckpoint
from wasp.retinaface.logger import build_mlflow
from wasp.retinaface.loss import LossWeights, MultiBoxLoss
from wasp.retinaface.model import RetinaFace
from wasp.retinaface.pipeline import RetinaFacePipeline
from wasp.retinaface.preprocess import preprocess
from wasp.retinaface.priors import priorbox

env = Env()
env.read_env()


def main(
    train_labels: str = None,
    valid_labels: str = None,
    resolution: tuple[int, int] = (840, 840),
    epochs: int = 20,
) -> None:
    pl.trainer.seed_everything(137)
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

    pipeline = RetinaFacePipeline(
        train_labels=train_labels or env.str("TRAIN_LABEL_PATH"),
        valid_labels=valid_labels or env.str("VALID_LABEL_PATH"),
        model=model,
        resolution=resolution,
        preprocessing=partial(preprocess, img_dim=resolution[0]),
        priorbox=priors,
        build_optimizer=partial(
            torch.optim.SGD,
            lr=0.001,
            weight_decay=0.0001,
            momentum=0.9,
        ),
        build_scheduler=partial(
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
            T_0=10,
            T_mult=2,
        ),
        loss=MultiBoxLoss(
            num_classes=2,
            overlap_thresh=0.35,
            neg_mining=True,
            neg_pos=7,
            neg_overlap=0.35,
            priors=priors,
            weights=LossWeights(
                localization=2,
                classification=1,
                landmarks=1,
            ),
        ),
    )

    Path("./retinaface-results").mkdir(
        exist_ok=True,
        parents=True,
    )

    trainer = pl.Trainer(
        # gpus=4,
        # amp_level=O1,
        max_epochs=epochs,
        # distributed_backend=ddp,
        num_sanity_val_steps=0,
        # progress_bar_refresh_rate=1,
        benchmark=True,
        precision=16,
        sync_batchnorm=True,
        logger=build_mlflow(),
        callbacks=[
            BestModelCheckpoint(
                monitor="mAP",
                verbose=True,
                mode="max",
                save_top_k=-1,
                save_weights_only=True,
            )
        ],
    )

    trainer.fit(pipeline)


if __name__ == "__main__":
    main()
