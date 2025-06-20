import os

import mlflow
import torch
from hydra.utils import instantiate
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf

from utils.misc import is_folder_filename_path
from utils.mlflow_utils import download_artifact, get_experiment_name_from_run


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
    run_id, checkpoint_tag = cfg.checkpoint.split("/", 1)

    # Download the config file from the training run to retrieve model architecture
    config_path = download_artifact(cfg, run_id, "config.yaml")
    train_cfg = OmegaConf.load(config_path)

    dataset_id = train_cfg.dataset_id
    dataset_name = get_experiment_name_from_run(dataset_id)

    # Model is determined by combination of model name and dataset 
    model_name = cfg.deploy_model_name
    mlflow.set_tag("mlflow.runName", f"deploy_{model_name}_{run_id}")

    # Download the checkpoint file
    checkpoint_path = download_artifact(cfg, run_id, f"{checkpoint_tag}_model.pt")
    
    # Retrieve model class
    model = instantiate(train_cfg.model.init, cfg=train_cfg, _recursive_=False)
    
    # FIX 1: Load the state dict into the model
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.nn.load_state_dict(state_dict)
    # FIX 2: Set model to evaluation mode
    model.nn.eval()
    
    # Create artifacts directory for this deployment
    artifacts_dir = os.path.join(cfg.runs_path, "deploy_artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # FIX 3: Save the loaded weights for explicit inclusion in the MLFlow model
    weights_path = os.path.join(artifacts_dir, "model_weights.pt")
    torch.save(state_dict, weights_path)
    
    # Add a proper load_context method to the model to ensure weights are loaded
    original_load_context = model.load_context
    
    def enhanced_load_context(self, context):
        # Call original method
        original_load_context(context)
        
        # Explicitly load weights if available
        if "model_weights" in context.artifacts:
            weights_path = context.artifacts["model_weights"]
            if os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location="cpu")
                self.nn.load_state_dict(state_dict)
                self.nn.eval()
    
    # Add the enhanced method to the model
    model.load_context = enhanced_load_context.__get__(model, type(model))

    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=model,
        code_paths=["models", "utils"],
        signature=model._signature,
        input_example=model._input_example,
        registered_model_name=model_name,
        pip_requirements=["-r requirements.txt"],
        artifacts={"model_weights": weights_path},  # FIX 4: Include weights as explicit artifact
        metadata={
            "checkpoint": cfg.checkpoint,
        }
    )

    client = MlflowClient(mlflow.get_tracking_uri())
    all_versions = client.search_model_versions(f"name = '{model_name}'")  
    model_info = max(all_versions, key=lambda v: int(v.version))

    client.set_model_version_tag(
        name=model_name,
        version=model_info.version,
        key='model',
        value=train_cfg.model.name
    )
    client.set_model_version_tag(
        name=model_name,
        version=model_info.version,
        key='dataset',
        value=dataset_name
    )
    client.set_model_version_tag(
        name=model_name,
        version=model_info.version,
        key='dataset_version',
        value=train_cfg.dataset_id
    )
    client.update_model_version(
        name=model_name,
        version=model_info.version,
        description=model._description
    )
    
    return model_info