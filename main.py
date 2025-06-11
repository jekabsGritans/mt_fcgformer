from datetime import datetime

import hydra
import mlflow
from hydra.utils import instantiate
from omegaconf import DictConfig

import utils.transforms as T
from datasets import MLFlowDatasetAggregator
from deploy import deploy_model_from_config
from train import Trainer
from utils.misc import is_folder_filename_path
from utils.mlflow_utils import (configure_mlflow_auth,
                                get_experiment_name_from_run)


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):

    configure_mlflow_auth()

    # dataset-specific transforms for training and evaluation
    train_transforms = T.Compose.from_hydra(cfg.train_transforms)
    eval_transforms = T.Compose.from_hydra(cfg.eval_transforms)

    # init model
    model = instantiate(cfg.model.init, cfg=cfg, _recursive_=False)
    nn = model.nn

    assert cfg.mode in ["train", "deploy"], f"Invalid mode: {cfg.mode}. Must be one of ['train', 'deploy']"

    # start MLflow run
    mlflow.set_experiment(cfg.experiment_name)

    # mode_model_dataset_timestamp
    dataset_name = get_experiment_name_from_run(cfg.dataset_id)
    cfg.dataset_name = dataset_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_name = f"{cfg.mode}_{cfg.model.name}_{dataset_name}_{timestamp}"
    with mlflow.start_run(run_name=run_name) as run:
        
        mlflow.set_tag("mode", cfg.mode)
        mlflow.set_tag("model", cfg.model.name)
        mlflow.set_tag("dataset", dataset_name)

        if cfg.mode == "train":
            train_agg = MLFlowDatasetAggregator(cfg=cfg, dataset_id=cfg.dataset_id, split="train", spectrum_transform=train_transforms)
            val_agg = MLFlowDatasetAggregator(cfg=cfg, dataset_id=cfg.dataset_id, split="valid", spectrum_transform=eval_transforms)

            # pos weights only relevant for training
            fg_pos_weights = train_agg.datasets["nist"].fg_pos_weights
            aux_pos_weights = train_agg.datasets["nist"].aux_pos_weights
            nn.setup_loss(fg_pos_weights=fg_pos_weights, aux_pos_weights=aux_pos_weights)

            trainer = Trainer(nn=nn, train_agg=train_agg, val_agg=val_agg, cfg=cfg)
            trainer.train()

        elif cfg.mode == "deploy":
            # Deploy model to MLflow registry
            assert cfg.checkpoint is not None, "Must provide checkpoint for deployment"
            deploy_model_from_config(cfg)
    
if __name__ == "__main__":
    main()