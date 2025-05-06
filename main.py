import os

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

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
    train_dataset = instantiate(cfg.dataset.train, transform=None)
    val_dataset = instantiate(cfg.dataset.valid, transform=None)

    pos_weights = train_dataset.get_pos_weights()
    model = instantiate(cfg.model, pos_weights=pos_weights)

    trainer = Trainer(model=model, train_dataset=train_dataset, val_dataset=val_dataset, device=cfg.device, **cfg.trainer)

    # start training
    trainer.train()

def run_test(cfg: DictConfig):
    # hydra auto-instantiates the model and dataset
    model = instantiate(cfg.model, pos_weights=None)
    eval_dataset = instantiate(cfg.dataset.test, transform=None)

    evaluator = Evaluator(model=model, eval_dataset=eval_dataset, **cfg.tester)

    # start evaluation
    evaluator.evaluate()

if __name__ == "__main__":
    main()