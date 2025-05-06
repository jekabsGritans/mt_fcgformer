import os

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

import utils.transforms as T
from eval import Evaluator
from train import Trainer
from utils.mlflow_utils import MLFlowManager


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    os.makedirs(cfg.run_path, exist_ok=True)

    mlflow_manager = MLFlowManager(cfg)

    if cfg.mode == "train":
        mlflow_manager.start_run(cfg.run_name)
        run_training(cfg, mlflow_manager)

    elif cfg.mode == "continue":
        assert cfg.run_id is not None, "Run ID must be specified for continue mode."

        mlflow_manager.continue_run(cfg.run_id)
        run_training(cfg, mlflow_manager, continue_training=True)

    elif cfg.mode == "test":
        # TODO: this should be handled differently because probablt needs to be associated with an experiment but not a run? 
        # mlflow_manager
        # run_test(cfg)
        ...
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}.")

def run_training(cfg: DictConfig, mlflow_manager: MLFlowManager, continue_training: bool = False):
    # hydra auto-instantiates the model and dataset

    train_transforms = T.Compose.from_hydra(cfg.dataset.train_transforms)
    train_dataset = instantiate(cfg.dataset.train, transform=train_transforms)
    train_dataset.to(cfg.device)

    val_transforms = T.Compose.from_hydra(cfg.dataset.eval_transforms)
    val_dataset = instantiate(cfg.dataset.valid, transform=val_transforms)
    val_dataset.to(cfg.device)

    pos_weights = train_dataset.get_pos_weights()
    model = instantiate(cfg.model, pos_weights=pos_weights)
    
    validator = Evaluator(
        model=model, 
        eval_dataset=val_dataset, 
        device=cfg.device, 
        mlflow_manager=mlflow_manager,
        **cfg.evaluator
    )

    trainer = Trainer(
        model=model, 
        train_dataset=train_dataset, 
        validator=validator, 
        device=cfg.device, 
        mlflow_manager=mlflow_manager,
        model_name=cfg.model_name,
        **cfg.trainer
    )

    trainer.train(continue_training=continue_training)

def run_test(cfg: DictConfig):
    # # hydra auto-instantiates the model and dataset
    # model = instantiate(cfg.model, pos_weights=None)

    # test_transforms = T.Compose.from_hydra(cfg.dataset.eval_transforms)
    # test_dataset = instantiate(cfg.dataset.test, transform=test_transforms)
    # test_dataset.to(cfg.device)

    # evaluator = Evaluator(model=model, eval_dataset=test_dataset, device=cfg.device, **cfg.evaluator)

    # # start evaluation
    # evaluator.evaluate()
    ...

if __name__ == "__main__":
    main()