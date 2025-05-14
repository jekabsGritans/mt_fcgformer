import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import utils.transforms as T
from datasets import MLFlowDataset
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
    train_transforms = T.Compose.from_hydra(cfg.train_transforms)
    eval_transforms = T.Compose.from_hydra(cfg.eval_transforms)


    # init model
    model = instantiate(cfg.model.init)

    if cfg.mode == "train":
        train_dataset = MLFlowDataset(dataset_id=cfg.dataest, split="train", transform=train_transforms)
        val_dataset = MLFlowDataset(dataset_id=cfg.dataset, split="val", transform=eval_transforms)

        # pos weights only relevant for training
        model.neural_net.set_pos_weights(train_dataset.pos_weights)

        trainer = Trainer(model=model, train_dataset=train_dataset, val_dataset=val_dataset, cfg=cfg)
        trainer.train()

    elif cfg.mode == "test":
        test_dataset = MLFlowDataset(dataset_id=cfg.dataset, split="test", transform=eval_transforms)
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