import os
import time

import mlflow
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import MLFlowDataset
from eval.metrics import (compute_exact_match_ratio, compute_overall_accuracy,
                          compute_per_class_accuracy)
from models import BaseModel
from utils.misc import dict_to_device, is_folder_filename_path
from utils.mlflow_utils import (download_artifact, get_run_id,
                                upload_sync_artifacts)


class Trainer:
    def __init__(self, model: BaseModel, train_dataset: MLFlowDataset, val_dataset: MLFlowDataset, cfg: DictConfig):
        self.model = model
        self.model.neural_net = self.model.neural_net.to(cfg.device)
        self.train_dataset = train_dataset.to(cfg.device)
        self.val_dataset = val_dataset.to(cfg.device)

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
            self.model.neural_net.parameters(), lr=cfg.trainer.lr
        )

        self.best_val_loss = float('inf')

        self.cfg = cfg

    def load_checkpoint(self, model_path: str, optim_path: str):
        """
        Load model and optimizer states from checkpoint.
        """

        self.model.neural_net.load_state_dict(torch.load(model_path))
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
        torch.save(self.model.neural_net.state_dict(), local_model_path)

        local_optim_path = os.path.join(self.cfg.runs_path, run_id, f"{tag}_optim.pt")
        torch.save(self.optimizer.state_dict(), local_optim_path)
    
    def validate(self, total_steps: int) -> tuple[float, float]:
        """
        Validate the model on the validation dataset.
        :return: Tuple of (average loss, average accuracy)
        """

        assert self.val_dataset.target is not None, "Validation dataset must have targets for evaluation."
        predictions = torch.zeros_like(self.val_dataset.target, device=self.cfg.device) # (num_samples, num_classes) 0/1 for each class

        self.model.neural_net.eval()

        val_loss = 0.0
        samples_seen = 0

        with torch.no_grad():
            start_idx = 0
            for batch in tqdm(self.val_loader, desc="Testing", unit="batch"):
                batch = dict_to_device(batch, self.cfg.device)
                step_out = self.model.step(batch)
                logits = step_out["logits"] 
                loss = step_out["loss"]

                batch_size = logits.shape[0]
                samples_seen += batch_size

                val_loss += loss.item() * batch_size

                preds = torch.sigmoid(logits)
                preds = (preds > 0.5).float()

                end_idx = start_idx + batch_size
                predictions[start_idx:end_idx] = preds

                start_idx = end_idx
        
        val_loss = val_loss / samples_seen if samples_seen > 0 else 0.0

        per_class_acc = compute_per_class_accuracy(predictions, self.val_dataset.target)
        overall_acc = compute_overall_accuracy(predictions, self.val_dataset.target)
        exact_match_ratio = compute_exact_match_ratio(predictions, self.val_dataset.target)

        # Log validation metrics
        metrics = {
            "val/loss": val_loss,
            "val/overall_accuracy": overall_acc,
            "val/exact_match_ratio": exact_match_ratio,
        }
        
        # Log per-class metrics
        for class_idx, acc in per_class_acc.items():
            class_name = self.val_dataset.target_names[class_idx]
            metrics[f"val/class_acc/{class_name}"] = acc
        
        mlflow.log_metrics(metrics, step=total_steps)  

        return val_loss, overall_acc

    def train(self):
        if self.cfg.checkpoint is not None:
            assert is_folder_filename_path(self.cfg.checkpoint), "Checkpoint path should be of form {run_id}/{filename}"
            run_id, filename = self.cfg.checkpoint.split("/")
            self.download_checkpoint(run_id, filename)

        self.best_val_loss = float("inf")
        total_steps = 0 # global step counter

        try:
            for epoch in range(self.cfg.trainer.epochs):
                self.model.neural_net.train()

                for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch [{epoch+1}/{self.cfg.trainer.epochs}]")):
                    batch = dict_to_device(batch, self.cfg.device)

                    step_out = self.model.step(batch)
                    loss = step_out["loss"]

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # log batch metrics
                    if (batch_idx + 1) % self.cfg.trainer.log_interval_steps == 0 or batch_idx == len(self.train_loader) - 1:
                        mlflow.log_metrics({
                            "train/batch_loss": loss.item(),
                            "train/epoch": epoch + 1,
                            "train/lr": self.optimizer.param_groups[0]["lr"],
                        }, step=total_steps)

                    total_steps += 1

                mlflow.log_metrics({"epoch_completed": epoch + 1}, step=total_steps)

                # Validation

                val_loss, val_acc = self.validate(total_steps=total_steps) 

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best")
                    self.save_checkpoint("latest")

                elif (epoch + 1) % self.cfg.trainer.checkpoint_interval_epochs == 0:
                    self.save_checkpoint(f"latest")
        finally:
            upload_sync_artifacts(self.cfg)