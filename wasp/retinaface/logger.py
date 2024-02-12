from environs import Env
from pytorch_lightning.loggers.mlflow import MLFlowLogger


def build_mlflow(
    rname=None,
    branch=None,
    commit=None,
    model=None,
) -> MLFlowLogger:
    env = Env()
    env.read_env()
    return MLFlowLogger(
        tracking_uri=env.str("MLFLOW_TRACKING_URI"),
        experiment_name=env.str("MLFLOW_EXPERIMENT_NAME"),
        run_name=rname or env.str("EXPERIMENT_LABEL"),
        tags={
            "branch": branch or env.str("BRANCH"),
            "commit": commit or env.str("COMMIT"),
            "model": model or env.str("MODEL"),
        },
    )
