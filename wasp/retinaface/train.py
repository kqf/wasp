from functools import partial
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from addict import Dict as Adict

from wasp.retinaface.model import RetinaFace
from wasp.retinaface.pipeline import Paths, RetinaFacePipeline
from wasp.retinaface.preprocess import Preproc
from wasp.retinaface.priors import priorbox


def main(
    config="wasp/retinaface/configs/default.yaml",
    paths: Paths | None = None,
    resolution: tuple[int, int] = (1024, 1024),
) -> None:
    with open(config) as f:
        config = Adict(yaml.load(f, Loader=yaml.SafeLoader))

    pl.trainer.seed_everything(config.seed)

    paths = paths or Paths()
    model = RetinaFace(
        name="Resnet50",
        pretrained=True,
        return_layers={
            "layer2": 1,
            "layer3": 2,
            "layer4": 3,
        },
        in_channels=256,
        out_channels=256,
    )

    pipeline = RetinaFacePipeline(
        config,
        paths,
        model=model,
        preprocessing=partial(Preproc, img_dim=resolution[0]),
        priorbox=priorbox(
            min_sizes=[[16, 32], [64, 128], [256, 512]],
            steps=[8, 16, 32],
            clip=False,
            image_size=resolution,
        ),
        build_optimizer=partial(
            torch.optim.SGD,
            lr=0.001,
            weight_decay=0.0001,
            momentum=0.9,
        ),
    )

    Path("./retinaface-results").mkdir(
        exist_ok=True,
        parents=True,
    )

    trainer = pl.Trainer(
        # gpus=4,
        # amp_level=O1,
        max_epochs=1,
        # distributed_backend=ddp,
        num_sanity_val_steps=0,
        # progress_bar_refresh_rate=1,
        benchmark=True,
        precision=16,
        sync_batchnorm=True,
    )

    trainer.fit(pipeline)


if __name__ == "__main__":
    main()
