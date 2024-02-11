from environs import Env
from pytorch_lightning.loggers.mlflow import MLFlowLogger


def build_mlflow(rname, branch, commit, model) -> MLFlowLogger:
    env = Env()
    env.read_env()
    return MLFlowLogger(
        tracking_uri=env.str("MLFLOW_TRACKING_URI"),
        experiment_name=env.str("MLFLOW_EXPERIMENT_NAME"),
        run_name=rname,
        tags={
            "branch": branch,
            "commit": commit,
            "model": model,
        },
    )
