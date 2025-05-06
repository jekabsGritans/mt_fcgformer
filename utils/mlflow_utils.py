import os

import mlflow
import torch
from omegaconf import DictConfig, OmegaConf


class MLFlowManager:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg 
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create experiment (or get existing one)
        mlflow.set_experiment(self.cfg.experiment_name)
    
    @property
    def run_id(self):
        assert mlflow.active_run() is not None, "No active MLflow run found. Please start a run first."
        return mlflow.active_run().info.run_id
    
    def start_run(self, run_name: str):
        """
        Setup MLflow tracking for experiment.
        
        Args:
            cfg: The Hydra config
            run_name: Optional run name (defaults to model name)
        """
        
        mlflow.start_run(run_name=run_name)
        
        # Log config parameters
        flat_config = OmegaConf.to_container(self.cfg, resolve=True)
        mlflow.log_params(self._flatten_dict(flat_config))

        # Log config as YAML artifact
        os.makedirs(self.cfg.run_path, exist_ok=True)
        config_path = os.path.join(self.cfg.run_path, "config.yaml")

        with open(config_path, "w") as f:
            OmegaConf.save(config=self.cfg, f=f)

        mlflow.log_artifact(config_path)

    def continue_run(self, run_id: str):
        """
        Continue an existing MLflow run.
        
        Args:
            run_id: The ID of the existing run
        """
        mlflow.start_run(run_id=run_id)

        # verify that config is identical to the one used in the original run. otherwise, raise an error
        self.download_artifacts()

        flat_config = OmegaConf.to_container(self.cfg, resolve=True)
        config_path = os.path.join(self.cfg.run_path, "config.yaml")
        with open(config_path, "r") as f:
            loaded_config = OmegaConf.load(f)
            loaded_flat_config = OmegaConf.to_container(loaded_config, resolve=True)
        if flat_config != loaded_flat_config:
            raise ValueError("Config does not match the one used in the original run. Please check your config file.")
       
    def _flatten_dict(self, d, parent_key='', sep='.'):
        """
        Flatten a nested dictionary for MLflow parameter logging
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (list, tuple)):
                # Convert lists to strings to avoid MLflow errors
                items.append((new_key, str(v)))
            elif not isinstance(v, (str, int, float, bool)) and v is not None:
                # Skip complex objects
                continue
            else:
                items.append((new_key, v))
        return dict(items)

    def log_metrics(self, metrics_dict: dict, step: int):
        """Log metrics to MLflow"""
        mlflow.log_metrics(metrics_dict, step=step)

    def save_final_model(self, model, model_name: str):
        """
        Log trained PyTorch model to MLflow (not for checkpointing).
        
        Args:
            model: PyTorch model
            model_name: Name for the logged model  # TODO: figure out if this should be per architecture or per experiment or idk
        """
        mlflow.pytorch.log_model(model, model_name)

    def end_run(self, ):
        """End the current MLflow run"""
        mlflow.end_run()

    def log_checkpoint(self, model, optimizer, epoch: int, best_val_loss: float, filename: str):
        """
        Save model checkpoint and log it as an MLflow artifact.
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            epoch: Current epoch number
            best_val_loss: Best validation loss so far 
            filename: Filename of the checkpoint (not full path)
        Returns:
            Path to the saved checkpoint
        """
        # Create checkpoint dictionary with all necessary info to resume training
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }
        
        # Save checkpoint locally
        local_path = os.path.join(self.cfg.run_path, "artifacts/checkpoints", filename)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        torch.save(checkpoint, local_path)

        # Log the checkpoint as an artifact in MLflow
        # this uses local_path to get file, but only references the filename in MLflow
        mlflow.log_artifact(local_path, artifact_path="artifacts/checkpoints") 
    
    def load_checkpoint(self, model, optimizer, filename: str) -> dict:
        """
        Load a model checkpoint from a local file.
        
        Args:
            model: PyTorch model to load weights into
            optimizer: PyTorch optimizer to load state into 
            filename: .{artifact_path}/checkpoints/{filename} will be loaded

        Returns:
            model: PyTorch model
            optimizer: PyTorch optimizer
            epoch: Current epoch number
            best_val_loss: Best validation loss so far 
        """

        # Check if checkpoint exists locally, if not try to download from MLflow
        local_path = os.path.join(self.cfg.run_path, "artifacts/checkpoints", filename)
        assert os.path.exists(local_path), f"Checkpoint file not found locally: {local_path}"
    
        # Load the checkpoint
        checkpoint = torch.load(local_path, map_location=self.cfg.device)

        model.load_state_dict(checkpoint['model_state_dict']).to(self.cfg.device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        out = {
            'model': model,
            'optimizer': optimizer,
            'epoch': checkpoint['epoch'],
            'val_loss': checkpoint['val_loss'],
            }
        return out
    
    def download_artifacts(self):
        """
        Download the latest artifacts from the current MLflow run.
        """

        # Get the current run ID
        run_id = mlflow.active_run().info.run_id

        run_artifact_path = os.path.join(self.cfg.run_path, "artifacts")

        # Download artifacts
        mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=run_artifact_path)