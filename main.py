import hydra
import mlflow
from hydra.utils import instantiate
from omegaconf import DictConfig

import utils.transforms as T
from datasets import MLFlowDataset
from deploy import deploy_model_from_config
from eval import Tester
from train import Trainer
from utils.misc import is_folder_filename_path
from utils.mlflow_utils import configure_mlflow_auth


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):

    configure_mlflow_auth()

    # dataset-specific transforms for training and evaluation
    train_transforms = T.Compose.from_hydra(cfg.train_transforms)
    eval_transforms = T.Compose.from_hydra(cfg.eval_transforms)

    # init model
    model = instantiate(cfg.model.init, cfg=cfg)
    nn = model.nn

    assert cfg.mode in ["train", "test", "deploy"], f"Invalid mode: {cfg.mode}. Must be one of ['train', 'test', 'deploy']"

    # start MLflow run
    mlflow.set_experiment(cfg.experiment_name)

    with mlflow.start_run(run_name=cfg.run_name) as run:
        if cfg.mode == "train":
            train_dataset = MLFlowDataset(cfg=cfg, dataset_id=cfg.dataset, split="train", transform=train_transforms)
            val_dataset = MLFlowDataset(cfg=cfg, dataset_id=cfg.dataset, split="valid", transform=eval_transforms)

            # pos weights only relevant for training
            nn.set_pos_weights(train_dataset.pos_weights)

            trainer = Trainer(nn=nn, train_dataset=train_dataset, val_dataset=val_dataset, cfg=cfg)
            trainer.train()

        elif cfg.mode == "test":
            test_dataset = MLFlowDataset(cfg=cfg, dataset_id=cfg.dataset, split="test", transform=eval_transforms)
            tester = Tester(nn=nn, test_dataset=test_dataset, cfg=cfg)
            tester.test()

        elif cfg.mode == "deploy":
            # Deploy model to MLflow registry
            assert cfg.checkpoint is not None, "Must provide checkpoint for deployment"
            deploy_model_from_config(cfg)
    
if __name__ == "__main__":
    main()