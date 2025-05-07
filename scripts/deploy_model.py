"""
Deploy a trained model from a checkpoint to the MLflow Model Registry.

This script:
1. Downloads the specified checkpoint from an MLflow run
2. Loads the config.yaml to determine proper model naming
3. Extracts only the model weights (no optimizer state)
4. Registers the model to MLflow Model Registry for deployment

Usage:
    python deploy_model.py --checkpoint=run_id/filename
    
Example:
    python deploy_model.py --checkpoint=1a2b3c4d/best_checkpoint.pt

NOTE: For efficiency, run this script somewhere on the deployment server to avoid download-upload.
"""

import argparse
import os
import sys

import mlflow
import torch
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from utils.misc import is_folder_filename_path

# Add project root to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import set_config
from utils.mlflow_utils import configure_mlflow_auth, download_artifact


def parse_args():
    parser = argparse.ArgumentParser(description='Deploy a model from checkpoint to MLflow Model Registry')
    parser.add_argument('--checkpoint', required=True, type=str, 
                        help='Checkpoint in format run_id/filename.pt')
    parser.add_argument('--model-version', type=str, default=None,
                        help='Version name for the model (defaults to run_id)')
    return parser.parse_args()


def download_config(run_id):
    """Download and load the config.yaml from the specified run."""
    config_path = download_artifact(run_id, "config.yaml")
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return OmegaConf.create(config_dict)


def deploy_model(checkpoint_path, run_id, model_version=None, stage=None):
    """
    Deploy a model from checkpoint to MLflow Model Registry.
    
    Args:
        checkpoint_path: Local path to the checkpoint file
        run_id: MLflow run ID
        model_version: Optional version name (defaults to run_id)
        stage: Optional stage for the model version
    """
    # Configure MLflow authentication
    configure_mlflow_auth()
    
    # Download original run config
    print(f"Downloading configuration from run {run_id}...")
    config = download_config(run_id)
    
    # Make config available globally to any utils that might need it
    set_config(config)
    
    # Determine model name from config
    model_name = f"{config.model.name}_{config.dataset.name}"
    print(f"Deploying model as: {model_name}")
    
    # Load checkpoint data
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
    
    # Instantiate model with correct architecture from config
    print("Instantiating model...")
    model = instantiate(config.model.init, pos_weights=None)
    
    # Load only the model weights
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()
    
    # Start a new MLflow run for deployment
    with mlflow.start_run(run_name=f"deploy_{model_name}"):
        # Log the original run ID as a tag
        mlflow.set_tag("original_run_id", run_id)
        
        # Log the model to the MLflow Model Registry
        print("Logging model to MLflow Model Registry...")
        model_info = mlflow.pytorch.log_model(
            model, 
            "model",
            registered_model_name=model_name
        )
        
        # Version information
        version = model_version or run_id
        mlflow.set_tag("version", version)
        
        # Optional: Add description and metadata
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version
        
        # Update description
        client.update_model_version(
            name=model_name,
            version=latest_version,
            description=f"Deployed from checkpoint {checkpoint_path} in run {run_id}"
        )
        
        print(f"Model {model_name} version {latest_version} successfully registered!")
        print(f"Model URI: {model_info.model_uri}")
        
        return model_info


def main():
    args = parse_args()
    
    if not is_folder_filename_path(args.checkpoint):
        print("Error: Checkpoint must be in format 'run_id/filename.pt'")
        sys.exit(1)
        
    run_id, filename = args.checkpoint.split("/")
    
    try:
        # Download the checkpoint
        print(f"Downloading checkpoint {filename} from run {run_id}...")
        local_path = download_artifact(run_id, filename)
        
        # Deploy the model
        model_info = deploy_model(
            checkpoint_path=local_path,
            run_id=run_id,
            model_version=args.model_version,
        )
        
        print("\nDeployment successful!")
        print(f"To use this model: mlflow.pytorch.load_model('{model_info.model_uri}')")
        
    except Exception as e:
        print(f"Error deploying model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()