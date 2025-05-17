import os

import mlflow
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import MLFlowDataset
from eval.metrics import compute_metrics
from models import NeuralNetworkModule
from utils.misc import dict_to_device, is_folder_filename_path
from utils.mlflow_utils import (download_artifact, get_run_id, log_config,
                                log_config_params, upload_sync_artifacts)


class Trainer:
    def __init__(self, nn: NeuralNetworkModule, train_dataset: MLFlowDataset, val_dataset: MLFlowDataset, cfg: DictConfig):
        self.nn = nn.to(cfg.device)
        self.train_dataset = train_dataset.to(cfg.device)
        self.val_dataset = val_dataset.to(cfg.device)

        # update target names if not set in cfg
        if cfg.target_names is None:
            cfg.target_names = self.train_dataset.target_names

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.trainer.batch_size, shuffle=cfg.trainer.shuffle,
            num_workers=cfg.trainer.num_workers, pin_memory=cfg.trainer.pin_memory,
            persistent_workers=cfg.trainer.persistent_workers
            )
    
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=cfg.trainer.batch_size, shuffle=False,
            num_workers=cfg.trainer.num_workers, pin_memory=cfg.trainer.pin_memory,
            persistent_workers=cfg.trainer.persistent_workers
            )
         
        self.optimizer = torch.optim.Adam(
            self.nn.parameters(), lr=cfg.trainer.lr
        )

        self.best_val_loss = float('inf')

        self.cfg = cfg

    def load_checkpoint(self, model_path: str, optim_path: str):
        """
        Load model and optimizer states from checkpoint.
        """

        self.nn.load_state_dict(torch.load(model_path))
        self.optimizer.load_state_dict(torch.load(optim_path))
    
    def download_checkpoint(self, run_id: str, tag: str):
        """
        Download (and load) the checkpoint file from MLFlow.
        """
        model_path = download_artifact(self.cfg, run_id, f"{tag}_model.pt")
        optim_path = download_artifact(self.cfg, run_id, f"{tag}_optim.pt")

        self.load_checkpoint(model_path, optim_path)

    def save_checkpoint(self, tag: str):
        """
        Save current model and optimizer states to a checkpoint file in local runs.
        Args:
            tag (str): e.g. "best" results in runs/best_optim.pt and runs/best_model.pt
        """
        run_id = get_run_id()

        local_model_path = os.path.join(self.cfg.runs_path, run_id, f"{tag}_model.pt")
        torch.save(self.nn.state_dict(), local_model_path)

        local_optim_path = os.path.join(self.cfg.runs_path, run_id, f"{tag}_optim.pt")
        torch.save(self.optimizer.state_dict(), local_optim_path)
    
    def validate(self, total_steps: int) -> float:
        """
        Validate the model on the validation dataset.
        :return: Tuple of (average loss, average accuracy)
        """

        assert self.val_dataset.target is not None, "Validation dataset must have targets for evaluation."
        predictions = torch.zeros_like(self.val_dataset.target, device=self.cfg.device) # (num_samples, num_targets) 0/1 for each target

        self.nn.eval()

        val_loss = 0.0
        samples_seen = 0

        with torch.no_grad():
            start_idx = 0
            for batch in tqdm(self.val_loader, desc="Testing", unit="batch"):
                batch = dict_to_device(batch, self.cfg.device)
                step_out = self.nn.step(batch)
                logits = step_out["logits"] 
                loss = step_out["loss"]

                batch_size = logits.shape[0]
                samples_seen += batch_size

                val_loss += loss.item() * batch_size

                preds = torch.sigmoid(logits)
                preds = (preds > self.cfg.trainer.validator.threshold).float()

                end_idx = start_idx + batch_size
                predictions[start_idx:end_idx] = preds

                start_idx = end_idx
        
        val_loss = val_loss / samples_seen if samples_seen > 0 else 0.0

        metrics = compute_metrics(predictions, self.val_dataset.target)

        mlflow_metrics = {
            "val/overall/accuracy": metrics["overall_accuracy"],
            "val/overall/precision": metrics["overall_precision"],
            "val/overall/recall": metrics["overall_recall"],
            "val/overall/f1": metrics["overall_f1"],
            "val/overall/emr": metrics["exact_match_ratio"],
            "val/loss": val_loss
        }

        for target_idx, target_name in enumerate(self.val_dataset.target_names):
            mlflow_metrics[f"val/per_target/accuracy/{target_name}"] = metrics["per_target_accuracy"][target_idx]
            mlflow_metrics[f"val/per_target/precision/{target_name}"] = metrics["per_target_precision"][target_idx]
            mlflow_metrics[f"val/per_target/recall/{target_name}"] = metrics["per_target_recall"][target_idx]
            mlflow_metrics[f"val/per_target/f1/{target_name}"] = metrics["per_target_f1"][target_idx]

        mlflow.log_metrics(mlflow_metrics, step=total_steps)

        return val_loss

    def train(self):

        log_config(self.cfg)
        log_config_params(self.cfg)

        if self.cfg.checkpoint is not None:
            assert is_folder_filename_path(self.cfg.checkpoint), "Checkpoint path should be of form {run_id}/{tag}"
            run_id, tag = self.cfg.checkpoint.split("/")
            self.download_checkpoint(run_id, tag)

        self.best_val_loss = float("inf")
        total_steps = 0 # global step counter

        try:
            for epoch in range(self.cfg.trainer.epochs):
                self.nn.train()

                for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch [{epoch+1}/{self.cfg.trainer.epochs}]")):
                    batch = dict_to_device(batch, self.cfg.device)

                    step_out = self.nn.step(batch)
                    loss = step_out["loss"]

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # log batch metrics
                    if (batch_idx + 1) % self.cfg.trainer.log_interval_steps == 0 or batch_idx == len(self.train_loader) - 1:
                        mlflow.log_metrics({
                            "train/loss": loss.item(),
                            "train/lr": self.optimizer.param_groups[0]["lr"],
                        }, step=total_steps)

                    total_steps += 1

                # Validation

                val_loss = self.validate(total_steps=total_steps) 

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best")
                    self.save_checkpoint("latest")

                elif (epoch + 1) % self.cfg.trainer.checkpoint_interval_epochs == 0:
                    self.save_checkpoint(f"latest")
        finally:
            upload_sync_artifacts(self.cfg)