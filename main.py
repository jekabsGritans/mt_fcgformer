import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import utils.transforms as T
from deploy import deploy_model_from_config
from eval import Tester
from train import Trainer
from utils.mlflow_utils import configure_mlflow_auth, start_run


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # auth to mlflow
    configure_mlflow_auth()

    # set up experiment tracking 
    # log config only if train/test, since for deployment the checkpoint run's config is relevant not this one.
    do_log_config = cfg.mode in ["train", "test"]
    start_run(cfg, log_config=do_log_config)

    # dataset-specific transforms for training and evaluation
    train_transforms = T.Compose.from_hydra(cfg.dataset.train_transforms)
    eval_transforms = T.Compose.from_hydra(cfg.dataset.eval_transforms)

    # init model
    model = instantiate(cfg.model.init, pos_weights=cfg.dataset.pos_weights)

    if cfg.mode == "train":
        train_dataset = instantiate(cfg.dataset.train, transform=train_transforms, pos_weights=cfg.dataset.pos_weights)
        val_dataset = instantiate(cfg.dataset.valid, transform=eval_transforms, pos_weights=cfg.dataset.pos_weights)
        trainer = Trainer(model=model, train_dataset=train_dataset, val_dataset=val_dataset, cfg=cfg)
        trainer.train()

    elif cfg.mode == "test":
        test_dataset = instantiate(cfg.dataset.test, transform=eval_transforms, pos_weights=cfg.dataset.pos_weights)
        tester = Tester(model=model, test_dataset=test_dataset, cfg=cfg)
        tester.test()

    elif cfg.mode == "deploy":
        # Deploy model to MLflow registry
        assert cfg.checkpoint is not None, "Must provide checkpoint for deployment"
        deploy_model_from_config(cfg)

    else:
        raise ValueError(f"Unknown mode: {cfg.mode}.")
    
if __name__ == "__main__":
    main()