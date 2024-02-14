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

        desired_path = os.path.join(self.dirpath, "best.pth")  # type: ignore
        os.makedirs(os.path.dirname(desired_path), exist_ok=True)
        if os.path.exists(self.best_model_path):
            checkpoint = torch.load(self.best_model_path)
            pl_module.load_state_dict(checkpoint["state_dict"])
        # else save last
        torch.save(pl_module.model.state_dict(), desired_path)  # type: ignore

        logger = trainer.logger
        if not isinstance(logger, MLFlowLogger):
            raise ValueError("Can't save model without MLFLOW logger")

        logger.experiment.log_artifact(
            run_id=logger.run_id,
            local_path=desired_path,
            artifact_path="checkpoints",
        )
