import os

import mlflow
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import BaseDataset
from eval.metrics import (compute_exact_match_ratio, compute_overall_accuracy,
                          compute_per_class_accuracy)
from models import BaseModel
from utils.misc import dict_to_device, is_folder_filename_path
from utils.mlflow_utils import (download_artifact, log_config, upload_artifact,
                                upload_model)


class Trainer:

    def __init__(self, model: BaseModel, train_dataset: BaseDataset, val_dataset: BaseDataset, cfg: DictConfig):

        self.model = model.to(cfg.device)
        self.train_dataset = train_dataset.to(cfg.device)
        self.val_dataset = val_dataset.to(cfg.device)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.trainer.bath_size, shuffle=cfg.trainer.shuffle,
            num_workers=cfg.trainer.num_workers, pin_memory=cfg.trainer.pin_memory,
            persistent_workers=cfg.trainer.persistent_workers
            )
    
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=cfg.trainer.bath_size, shuffle=False,
            num_workers=cfg.trainer.num_workers, pin_memory=cfg.trainer.pin_memory,
            persistent_workers=cfg.trainer.persistent_workers
            )
         
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.trainer.lr
        )

        self.best_val_loss = float('inf')

        self.cfg = cfg

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model and optimizer states from checkpoint.
        Args:
            checkpoint_path (str): Local path to the checkpoint file.
        """

        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    def download_checkpoint(self, run_id: str, filename: str):
        """
        Download (and load) the checkpoint file from MLFlow.
        Args:
            run_id (str): MLFlow run ID.
            filename (str): Name of the checkpoint file (e.g. "latest_model.pt")
        """
        download_artifact(self.cfg, run_id, filename)
        self.load_checkpoint(filename)

    def upload_checkpoint(self, filename: str):
        """
        Save current model and optimizer states to a checkpoint file and upload to MLFlow.
        Args:
            filename (str): Name of the checkpoint file (e.g. "latest_model.pt")
        """

        local_path = os.path.join(self.cfg.run_path, filename)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, local_path)
        upload_artifact(local_path)
    
    def upload_best_model(self):
        """
        Save the best model checkpoint as deployable model to MLFlow.
        """
        raise NotImplementedError() # use upload_model() 

    def validate(self, epoch: int) -> tuple[float, float]:
        """
        Validate the model on the validation dataset.
        :return: Tuple of (average loss, average accuracy)
        """
        self.model.eval()

        assert self.val_dataset.target is not None, "Validation dataset must have targets for evaluation."
        predictions = torch.zeros_like(self.val_dataset.target, device=self.cfg.device) # (num_samples, num_classes) 0/1 for each class

        self.model.eval()

        val_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            start_idx = 0
            for batch in tqdm(self.val_loader, desc="Testing", unit="batch"):
                batch = dict_to_device(batch, self.cfg.device)
                step_out = self.model.step(batch)
                logits = step_out["logits"] 
                loss = step_out["loss"]

                batch_size = preds.shape[0]
                total_samples += batch_size

                val_loss += loss.item() * batch_size

                preds = torch.sigmoid(logits)
                preds = (preds > 0.5).float()

                end_idx = start_idx + batch_size
                predictions[start_idx:end_idx] = preds

                start_idx = end_idx
        
        val_loss = val_loss / total_samples if total_samples > 0 else 0.0

        per_class_acc = compute_per_class_accuracy(predictions, self.val_dataset.target)
        overall_acc = compute_overall_accuracy(predictions, self.val_dataset.target)
        exact_match_ratio = compute_exact_match_ratio(predictions, self.val_dataset.target)

        val_metrics = {
            "validation_loss": val_loss,
            "overall_accuracy": overall_acc,
            "per_class_accuracy": per_class_acc,
            "exact_match_ratio": exact_match_ratio,
        }

        # TODO: log validation metrics

        return val_loss, overall_acc

    def train(self):

        with mlflow.start_run(run_name=self.cfg.run_name):

            log_config(self.cfg)

            if self.cfg.checkpoint is not None:
                assert is_folder_filename_path(self.cfg.checkpoint), "Checkpoint path should be of form {run_id}/{filename}"
                run_id, filename = self.cfg.checkpoint.split("/")
                self.download_checkpoint(self.cfg.checkpoint_run_id, self.cfg.checkpoint.filename)

            self.best_val_loss = float("inf")

            for epoch in range(self.cfg.epochs):
                self.model.train()
                for batch in tqdm(self.train_loader, desc=f"Epoch [{epoch+1}/{self.cfg.epochs}]"):
                    batch = dict_to_device(batch, self.cfg.device)

                    step_out = self.model.step(batch)
                    loss = step_out["loss"]

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # TODO: log training metrics

                # Validation

                val_loss, val_acc = self.validate(epoch) 

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.upload_checkpoint("best_checkpoint.pt")
                    self.upload_checkpoint("latest_checkpoint.pt")

                elif (epoch + 1) % self.cfg.checkpoint_every == 0:
                        self.upload_checkpoint(f"latest_checkpoint.pt")

            self.upload_best_model()