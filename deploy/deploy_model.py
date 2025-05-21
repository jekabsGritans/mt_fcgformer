import mlflow
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

    dataset_id = train_cfg.dataset
    dataset_name = get_experiment_name_from_run(dataset_id)

    # Model is determined by combination of model name and dataset 
    model_name = f"{train_cfg.model.name}_{dataset_name}"
    mlflow.set_tag("mlflow.runName", f"deploy_{model_name}_{run_id}")

    # Retrieve model class
    model = instantiate(train_cfg.model.init, cfg=train_cfg, _recursive_=False)

    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=model,
        code_paths=["models", "utils"],
        signature=model._signature,
        input_example=model._input_example,
        registered_model_name=model_name,
        pip_requirements=["-r requirements.txt"],
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
        value=train_cfg.dataset
    )
    client.update_model_version(
        name=model_name,
        version=model_info.version,
        description=model._description
    )
