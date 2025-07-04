from __future__ import annotations

import os
import urllib.parse

import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, ListConfig, OmegaConf


def upload_sync_artifacts(cfg: DictConfig) -> int:
    """
    Upload all artifacts in the current local run path to MLflow.
    This is useful for synchronizing local files with MLflow tracking server.
    
    Returns:
        int: Number of files uploaded
    """
    run_id = get_run_id()
    
    # Get local run path
    local_run_path = os.path.join(cfg.runs_path, run_id)
    
    if not os.path.isdir(local_run_path):
        print(f"Run path does not exist: {local_run_path}")
        return 0
        
    print(f"Synchronizing artifacts from {local_run_path} to MLflow run {run_id}...")
    
    # Track number of files uploaded
    uploaded_count = 0
    
    # Get all files in the directory (no subdirectories)
    files = [f for f in os.listdir(local_run_path) if os.path.isfile(os.path.join(local_run_path, f))]
    
    for file in files:
        file_path = os.path.join(local_run_path, file)
        try:
            print(f"Uploading {file}...")
            mlflow.log_artifact(file_path)
            uploaded_count += 1
        except Exception as e:
            print(f"Error uploading {file}: {e}")
    
    print(f"Successfully uploaded {uploaded_count} artifacts to MLflow run {run_id}")
    
    return uploaded_count

def get_run_id() -> str:
    """
    Get the current MLflow run ID.
    If no active run is found, raise an error.
    """
    run = mlflow.active_run()
    if run is None:
        raise RuntimeError("No active MLflow run found. Please start a run first.")
    return run.info.run_id

def get_experiment_name_from_run(run_id: str) -> str:
    client = MlflowClient()
    run = client.get_run(run_id)
    experiment_id = run.info.experiment_id
    experiment = client.get_experiment(experiment_id)
    return experiment.name


def configure_mlflow_auth():
    """
    Configure MLflow authentication using environment variables.
    This allows connecting to remote MLflow servers that require authentication.
    """
    # Get auth credentials from environment variables
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")

    
    # Check if authentication is already in the URI
    parsed_uri = urllib.parse.urlparse(tracking_uri)
    if parsed_uri.username or '@' in parsed_uri.netloc:
        # Auth already present, don't modify
        mlflow.set_tracking_uri(tracking_uri)
        return
        
    # If credentials are provided, add them to the tracking URI
    assert username and password, "MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD must be set for authentication"

    # Only add auth for http/https URLs
    if parsed_uri.scheme in ('http', 'https'):
        netloc = f"{urllib.parse.quote(username)}:{urllib.parse.quote(password)}@{parsed_uri.netloc}"
        tracking_uri = parsed_uri._replace(netloc=netloc).geturl()
        
    # Set the tracking URI for MLflow
    mlflow.set_tracking_uri(tracking_uri)
    
def log_config(cfg: DictConfig) -> None:
    """
    Log the configuration to MLflow as a YAML artifact.
    This is useful for tracking the configuration used for each run.
    
    Args:
        cfg (DictConfig): Configuration object from Hydra
    """
    run_id = get_run_id()
    run_path = os.path.join(cfg.runs_path, run_id)
    os.makedirs(run_path, exist_ok=True)

    config_path = os.path.join(run_path, "config.yaml")

    with open(config_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)

    upload_artifact(config_path) # saves to runs:/{current_run_id}/config.yaml

def log_config_params(cfg: DictConfig) -> None:
    """
    Log the configuration parameters to MLflow as parameters.
    This is useful for tracking the parameters used for each run.
    
    Args:
        cfg (DictConfig): Configuration object from Hydra
    """
    params = _flatten_dict(cfg)
    mlflow.log_params(params)   

    

def download_artifact(cfg: DictConfig, run_id: str, filename: str) -> str:
    """
    Download an artifact from a specific MLflow run.

    Args:
        run_id (str): MLflow run ID.
        filename (str): Name of the artifact file (e.g. "latest_checkpoint.pt").
    Returns:
        str: Local path to the downloaded artifact.
    """
    dst_dir = os.path.join(cfg.runs_path, run_id)

    os.makedirs(dst_dir, exist_ok=True)

    local_path = os.path.join(dst_dir, filename)
    if os.path.isfile(local_path):
        print(f"Artifact {filename} already exists in {dst_dir}. Skipping download.")
        return local_path
    try:
        mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=filename, dst_path=dst_dir) # type: ignore
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


def _flatten_dict(d, parent_key='', sep='.'):
    """
    Flatten a nested dictionary, including OmegaConf DictConfig and ListConfig,
    for MLflow parameter logging.
    Lists containing dictionaries or other lists are processed element by element.
    Dictionaries within lists are flattened with indexed keys.
    Simple lists (of primitives) and other complex objects are converted to strings.
    """
    items = []
    # d is expected to be a dict or DictConfig.
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, (DictConfig, dict)):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, (ListConfig, list, tuple)):
            # Check if the list contains complex structures (dicts or other lists)
            # that need individual processing.
            contains_complex_structure = False
            if v:  # Only check if list is not empty
                for item_check in v:
                    if isinstance(item_check, (DictConfig, dict, ListConfig, list, tuple)):
                        contains_complex_structure = True
                        break
            
            if contains_complex_structure:
                for i, item in enumerate(v):
                    item_key = f"{new_key}{sep}{i}"
                    if isinstance(item, (DictConfig, dict)):
                        items.extend(_flatten_dict(item, item_key, sep=sep).items())
                    elif isinstance(item, (ListConfig, list, tuple)):
                        # For lists/tuples nested deeper, convert to string.
                        # This avoids overly complex keys or too many parameters.
                        items.append((item_key, str(item)))
                    elif not isinstance(item, (str, int, float, bool)) and item is not None:
                        # Other non-primitive, non-dict/list complex objects within a list: convert to string.
                        items.append((item_key, str(item)))
                    else:
                        # Primitives (or None) within a list that is being flattened.
                        items.append((item_key, item))
            else:
                # List is empty or contains only primitives (or complex objects that will be stringified).
                # Convert the entire list to its string representation.
                items.append((new_key, str(v)))
        elif not isinstance(v, (str, int, float, bool)) and v is not None:
            # For other complex objects (not dicts, not lists, not primitives), convert to string.
            # This replaces the original behavior of skipping them.
            items.append((new_key, str(v)))
        else:
            # Primitives (str, int, float, bool) or None.
            # MLflow's log_param handles None by not logging the parameter.
            items.append((new_key, v))
            
    return dict(items)