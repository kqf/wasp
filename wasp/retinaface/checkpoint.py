import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.mlflow import MLFlowLogger


class BestModelCheckpoint(ModelCheckpoint):
    def on_train_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        super().on_train_end(trainer, pl_module)
        if trainer.global_rank != 0:
            return

        local = f"{self.dirpath}/best.pth"
        os.makedirs(os.path.dirname(local), exist_ok=True)

        checkpoint = pl_module.model
        if os.path.exists(self.best_model_path):
            checkpoint = torch.load(self.best_model_path)

        torch.save(checkpoint.state_dict(), local)  # type: ignore
        torch.save(checkpoint, local.replace("best.pth", "full-best.pth"))

        logger = trainer.logger
        if not isinstance(logger, MLFlowLogger):
            raise ValueError("Can't save model without MLFLOW logger")

        logger.experiment.log_artifact(
            run_id=logger.run_id,
            local_path=local,
            artifact_path="checkpoints",
        )
