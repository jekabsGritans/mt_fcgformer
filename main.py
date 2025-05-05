import argparse
import os

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from deploy import Predictor
from eval import Evaluator
from train import Trainer
from utils.transforms import np_to_torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test", "predict"], required=True)
    return parser.parse_known_args()  # allow Hydra to parse the rest

# Parse user-defined args *before* Hydra takes over
args, _ = parse_args()

@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    if args.mode == "train":
        run_training(cfg)
    elif args.mode == "test":
        run_test(cfg)
    elif args.mode == "predict":
        run_prediction(cfg)

def run_training(cfg: DictConfig):
    # hydra auto-instantiates the model and dataset
    model = instantiate(cfg.model)
    train_dataset = instantiate(cfg.dataset.train)
    val_dataset = instantiate(cfg.dataset.valid)

    trainer = Trainer(model=model, train_dataset=train_dataset, val_dataset=val_dataset, **cfg.trainer)

    # start training
    trainer.train()


def run_test(cfg: DictConfig):
    # hydra auto-instantiates the model and dataset
    model = instantiate(cfg.model)
    eval_dataset = instantiate(cfg.dataset.test)

    evaluator = Evaluator(model=model, eval_dataset=eval_dataset, **cfg.tester)

    # start evaluation
    evaluator.evaluate()

def run_prediction(cfg: DictConfig):
    # hydra auto-instantiates the model 
    model = instantiate(cfg.model)

    predictor = Predictor(model=model, device=cfg.device, class_names=cfg.dataset.class_names, transform=np_to_torch)

    if not os.path.exists(cfg.predictor.sample) or cfg.predictor.sample[-4:] != ".npy":
        raise ValueError("Sample path must be an existing .npy file")

    # Load the sample spectrum
    sample_spectrum = np.load(cfg.predictor.sample)
    if sample_spectrum.ndim != 1:
        raise ValueError(f"Sample spectrum must be 1d. Got {sample_spectrum.ndim}d.")

    # Make the prediction   
    predictor.predict(sample_spectrum, threshold=cfg.predictor.threshold, plot=cfg.predictor.plot)


if __name__ == "__main__":
    main()