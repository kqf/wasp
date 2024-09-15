from functools import partial
from pathlib import Path

import pytorch_lightning as pl
import torch
from environs import Env

# from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.callbacks import TQDMProgressBar

from wasp.retinaface.checkpoint import BestModelCheckpoint

# from wasp.retinaface.closs import DetectionLoss
from wasp.retinaface.logger import build_mlflow
from wasp.retinaface.loss import MultiBoxLoss

# from wasp.retinaface.monitor import PyTorchGpuMonitorCallback
from wasp.retinaface.pipeline import RetinaFacePipeline
from wasp.retinaface.preprocess import preprocess

# from wasp.retinaface.priors import priorbox
from wasp.retinaface.ssd import SSDPure, build_priors

# from wasp.retinaface.model import RetinaFace


# from gpumonitor.callbacks.lightning import PyTorchGpuMonitorCallback


env = Env()
env.read_env()


def main(
    train_labels: str = None,
    valid_labels: str = None,
    resolution: tuple[int, int] = (640, 640),
    epochs: int = 20,
) -> None:
    pl.trainer.seed_everything(137)
    # model = RetinaFace(
    #     name="Resnet50",
    #     pretrained=True,
    #     return_layers={
    #         "layer2": 1,
    #         "layer3": 2,
    #         "layer4": 3,
    #     },
    #     out_channels=256,
    # )
    # model = build_model()
    model = SSDPure(resolution, n_classes=2)

    # priors = priorbox(
    #     min_sizes=[[16, 32], [64, 128], [256, 512]],
    #     steps=[8, 16, 32],
    #     clip=False,
    #     image_size=resolution,
    # )
    priors = build_priors(resolution)

    pipeline = RetinaFacePipeline(
        train_labels=train_labels or env.str("TRAIN_LABEL_PATH"),
        valid_labels=valid_labels or env.str("VALID_LABEL_PATH"),
        model=model,
        resolution=resolution,
        preprocessing=partial(preprocess, img_dim=resolution[0]),
        priorbox=priors,
        build_optimizer=partial(
            torch.optim.Adam,
            # lr=0.0001,
            # weight_decay=0.0001,
            # momentum=0.9,
        ),
        build_scheduler=partial(
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
            T_0=10,
            T_mult=2,
        ),
        # 64, n_pos=8
        loss=MultiBoxLoss(priors=priors),
        # 48, n_pos=8
        # loss=DetectionLoss(anchors=priors),
        # prepare_outputs=model.prepare_outputs,
    )

    Path("./retinaface-results").mkdir(
        exist_ok=True,
        parents=True,
    )

    trainer = pl.Trainer(
        # gpus=4,
        # amp_level=O1,
        # devices=8,
        max_epochs=epochs,
        strategy="ddp_find_unused_parameters_true",
        num_sanity_val_steps=0,
        benchmark=True,
        precision=16,
        sync_batchnorm=torch.cuda.is_available(),
        logger=build_mlflow(),
        callbacks=[
            BestModelCheckpoint(
                monitor="mAP",
                verbose=True,
                mode="max",
                save_top_k=-1,
                save_weights_only=True,
            ),
            TQDMProgressBar(
                refresh_rate=100,
            ),
            # DeviceStatsMonitor(), ~
            # PyTorchGpuMonitorCallback(delay=0.5, log_per_batch=True),
        ],
    )

    trainer.fit(pipeline)


if __name__ == "__main__":
    main()
