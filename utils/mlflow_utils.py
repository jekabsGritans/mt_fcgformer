from __future__ import annotations

import os

import mlflow
from omegaconf import OmegaConf

from utils.config import get_config


def get_run_id() -> str:
    """
    Get the current MLflow run ID.
    If no active run is found, raise an error.
    """
    run = mlflow.active_run()
    if run is None:
        raise RuntimeError("No active MLflow run found. Please start a run first.")
    return run.info.run_id

def setup_mlflow() -> None:
    """
    Setup MLflow tracking for experiment.
    1. Connect to MLflow server
    2. Create experiment (or get existing one)
    3. Start a run
    4. Log config for reproducibility
    """

    cfg = get_config()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    # Create experiment (or get existing one) 
    mlflow.set_experiment(cfg.experiment_name)

    # Start a run
    mlflow.start_run(run_name=cfg.run_name)

    # Log config

    ## as parameters
    flat_config = OmegaConf.to_container(cfg, resolve=True)
    mlflow.log_params(_flatten_dict(flat_config))

    ## as YAML artifact for reproducibility
    config_path = os.path.join(cfg.runs_path, "config.yaml")

    with open(config_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)

    upload_artifact(config_path) # saves to runs:/{current_run_id}/config.yaml

def download_artifact(run_id: str, filename: str) -> str:
    """
    Download an artifact from a specific MLflow run.

    Args:
        run_id (str): MLflow run ID.
        filename (str): Name of the artifact file (e.g. "latest_checkpoint.pt").
    Returns:
        str: Local path to the downloaded artifact.
    """
    cfg = get_config()
    dst_dir = os.path.join(cfg.runs_path, run_id)

    os.makedirs(dst_dir, exist_ok=True)

    try:
        mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=filename, dst_path=dst_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to download artifact {filename} from run {run_id}: {e}")

    return os.path.join(dst_dir, filename)

def upload_artifact(file_path: str) -> None:
    """
    Log an artifact to MLflow.
    Artifact name will be the same as the filename.
    Run id is active run id.

    Args:
        file_path (str): Local path to the file to log.
    """

    assert os.path.isfile(file_path), f"Path is not a file: {file_path}"
    assert mlflow.active_run() is not None, "No active MLflow run found. Please start a run first."

    mlflow.log_artifact(file_path)

def upload_model(model, model_name: str) -> None:
    """
    Upload final trained model to MLflow for versioning and deployment.

    Args:
        model: The trained model to save.
        model_name (str): Name of the model.
    """
    raise NotImplementedError()

def _flatten_dict(d, parent_key='', sep='.'):
    """
    Flatten a nested dictionary for MLflow parameter logging
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, (list, tuple)):
            # Convert lists to strings to avoid MLflow errors
            items.append((new_key, str(v)))
        elif not isinstance(v, (str, int, float, bool)) and v is not None:
            # Skip complex objects
            continue
        else:
            items.append((new_key, v))
    return dict(items)