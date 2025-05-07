import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import utils.transforms as T
from eval import Tester
from train import Trainer
from utils.mlflow_utils import log_config, setup_mlflow


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    os.makedirs(cfg.run_path, exist_ok=True)

    # setup mlflow
    setup_mlflow(cfg)

    # dataset-specific transforms for training and evaluation
    train_transforms = T.Compose.from_hydra(cfg.dataset.train_transforms)
    eval_transforms = T.Compose.from_hydra(cfg.dataset.eval_transforms)

    # init model
    model = instantiate(cfg.model, pos_weights=cfg.dataset.pos_weights)

    if cfg.mode == "train":
        train_dataset = instantiate(cfg.dataset.train, transform=train_transforms, pos_weights=cfg.dataset.pos_weights)
        val_dataset = instantiate(cfg.dataset.valid, transform=eval_transforms)
        trainer = Trainer(model=model, train_dataset=train_dataset, val_dataset=val_dataset, cfg=cfg)
        trainer.train()

    elif cfg.mode == "test":
        test_dataset = instantiate(cfg.dataset.test, transform=eval_transforms)
        tester = Tester(model=model, test_dataset=test_dataset, cfg=cfg)
        tester.test()

    else:
        raise ValueError(f"Unknown mode: {cfg.mode}.")
    
if __name__ == "__main__":
    main()