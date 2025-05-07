"""
Deploy a trained model from a checkpoint to the MLflow Model Registry.

This module:
1. Uses the specified checkpoint from an MLflow run
2. Extracts only the model weights (no optimizer state)
3. Registers the model to MLflow Model Registry for deployment

This is designed to be called from main.py with mode="deploy"
"""

import mlflow
import omegaconf
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from utils.misc import is_folder_filename_path
from utils.mlflow_utils import download_artifact, upload_artifact


def deploy_model_from_config(cfg: DictConfig):
    """
    Deploy a model from checkpoint specified in config to MLflow Model Registry.
    
    Args:
        cfg: Configuration from Hydra
    
    Returns:
        Model info from MLflow registry
    """
    # Ensure checkpoint is provided
    assert cfg.checkpoint is not None, "Must provide checkpoint for deployment"
    assert is_folder_filename_path(cfg.checkpoint), "Checkpoint must be in format 'run_id/filename.pt'"
    
    # Parse run_id and filename from checkpoint
    run_id, filename = cfg.checkpoint.split("/", 1)

    # Download the config file from the training run
    print(f"Downloading config from run {run_id}...")
    config_path = download_artifact(run_id, "config.yaml")
    train_cfg = OmegaConf.load(config_path)

    # Log the checkpoint id
    mlflow.set_tag("from_checkpoint", cfg.checkpoint)

    # Determine model name
    model_name = f"{train_cfg.model.name}_{train_cfg.dataset.name}"

    # Rename the run to reflect trained model name
    mlflow.set_tag("mlflow.runName", f"deploy_{model_name}_{run_id}")
   
    # Download the checkpoint
    print(f"Downloading checkpoint {filename} from run {run_id}...")
    checkpoint_path = download_artifact(run_id, filename)
    
    # Load checkpoint data
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu")

    # Instantiate model with correct architecture from config
    print("Instantiating model...")
    model = instantiate(train_cfg.model.init, pos_weights=None)
    
    # Load only the model weights
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()
    
    # Log the model to the MLflow Model Registry
    print("Logging model to MLflow Model Registry...")
    model_info = mlflow.pytorch.log_model(
        model, 
        "model",
        registered_model_name=model_name
    )
    
    # Optional: Add description and metadata using the MLflow client
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(model_name, stages=["None"])
    if versions:
        latest_version = versions[0].version
        
        # Update description
        client.update_model_version(
            name=model_name,
            version=latest_version,
            description=f"Deployed from checkpoint {cfg.checkpoint}"
        )
        
        print(f"Model {model_name} version {latest_version} successfully registered!")
    else:
        print(f"Model {model_name} successfully registered!")
    
    print(f"Model URI: {model_info.model_uri}")
    return model_info