import os

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

import utils.transforms as T
from eval import Evaluator
from train import Trainer


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        run_training(cfg)
    elif cfg.mode == "test":
        run_test(cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}.")

def run_training(cfg: DictConfig):
    # hydra auto-instantiates the model and dataset

    train_transforms = T.Compose.from_hydra(cfg.dataset.train_transforms)
    train_dataset = instantiate(cfg.dataset.train, transform=train_transforms)
    train_dataset.to(cfg.device)

    val_transforms = T.Compose.from_hydra(cfg.dataset.eval_transforms)
    val_dataset = instantiate(cfg.dataset.valid, transform=val_transforms)
    val_dataset.to(cfg.device)

    pos_weights = train_dataset.get_pos_weights()
    model = instantiate(cfg.model, pos_weights=pos_weights)
    
    validator = Evaluator(model=model, eval_dataset=val_dataset, device=cfg.device, **cfg.evaluator)

    trainer = Trainer(model=model, train_dataset=train_dataset, validator=validator, device=cfg.device, **cfg.trainer)

    # start training
    trainer.train()

def run_test(cfg: DictConfig):
    # hydra auto-instantiates the model and dataset
    model = instantiate(cfg.model, pos_weights=None)

    test_transforms = T.Compose.from_hydra(cfg.dataset.eval_transforms)
    test_dataset = instantiate(cfg.dataset.test, transform=test_transforms)
    test_dataset.to(cfg.device)

    evaluator = Evaluator(model=model, eval_dataset=test_dataset, device=cfg.device, **cfg.evaluator)

    # start evaluation
    evaluator.evaluate()

if __name__ == "__main__":
    main()