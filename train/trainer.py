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

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optimizer, T_0=cfg.trainer.scheduler_t0, T_mult=cfg.trainer.scheduler_tmult)

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
        Reports metrics for functional groups and auxiliary targets if they exist.
        :return: Average validation loss
        """
        # Initialize tensors for predictions
        fg_predictions = torch.zeros_like(self.val_dataset.fg_targets, device=self.cfg.device)
        
        # Initialize aux prediction tensors only if they exist
        if hasattr(self.val_dataset, 'aux_bool_targets') and self.val_dataset.aux_bool_targets.shape[1] > 0:
            aux_bool_predictions = torch.zeros_like(self.val_dataset.aux_bool_targets, device=self.cfg.device)
        else:
            aux_bool_predictions = None
            
        if hasattr(self.val_dataset, 'aux_float_targets') and self.val_dataset.aux_float_targets.shape[1] > 0:
            aux_float_predictions = torch.zeros_like(self.val_dataset.aux_float_targets, device=self.cfg.device)
        else:
            aux_float_predictions = None

        self.nn.eval()

        total_loss = 0.0
        fg_loss_sum = 0.0
        aux_bool_loss_sum = 0.0
        aux_float_loss_sum = 0.0
        samples_seen = 0

        with torch.no_grad():
            start_idx = 0
            for batch in tqdm(self.val_loader, desc="Validating", unit="batch"):
                batch = dict_to_device(batch, self.cfg.device)
                step_out = self.nn.step(batch)
                
                # Get losses
                batch_size = step_out["fg_logits"].shape[0]
                samples_seen += batch_size
                
                # Get total loss and component losses
                total_loss += step_out["loss"].item() * batch_size
                fg_loss_sum += step_out["fg_loss"].item() * batch_size
                
                # Get auxiliary losses if they exist
                if "aux_bool_loss" in step_out:
                    aux_bool_loss_sum += step_out["aux_bool_loss"].item() * batch_size
                    
                if "aux_float_loss" in step_out:
                    aux_float_loss_sum += step_out["aux_float_loss"].item() * batch_size
                
                # Make predictions for functional groups
                fg_logits = step_out["fg_logits"]
                fg_preds = torch.sigmoid(fg_logits)
                fg_preds = (fg_preds > self.cfg.trainer.validator.threshold).float()
                
                end_idx = start_idx + batch_size
                fg_predictions[start_idx:end_idx] = fg_preds
                
                # Make predictions for auxiliary boolean targets
                if "aux_bool_logits" in step_out and aux_bool_predictions is not None:
                    aux_bool_logits = step_out["aux_bool_logits"]
                    aux_bool_preds = torch.sigmoid(aux_bool_logits)
                    aux_bool_preds = (aux_bool_preds > self.cfg.trainer.validator.threshold).float()
                    aux_bool_predictions[start_idx:end_idx] = aux_bool_preds
                
                # Make predictions for auxiliary float targets
                if "aux_float_preds" in step_out and aux_float_predictions is not None:
                    aux_float_preds = step_out["aux_float_preds"]
                    aux_float_predictions[start_idx:end_idx] = aux_float_preds
                    
                start_idx = end_idx
        
        # Calculate average losses
        avg_total_loss = total_loss / samples_seen if samples_seen > 0 else 0.0
        avg_fg_loss = fg_loss_sum / samples_seen if samples_seen > 0 else 0.0
        avg_aux_bool_loss = aux_bool_loss_sum / samples_seen if samples_seen > 0 else 0.0
        avg_aux_float_loss = aux_float_loss_sum / samples_seen if samples_seen > 0 else 0.0
        
        # Initialize metrics dictionary for MLflow
        mlflow_metrics = {
            "val/loss/total": avg_total_loss,
            "val/loss/fg": avg_fg_loss
        }
        
        # Add auxiliary losses if they exist
        if aux_bool_loss_sum > 0:
            mlflow_metrics["val/loss/aux_bool"] = avg_aux_bool_loss
            
        if aux_float_loss_sum > 0:
            mlflow_metrics["val/loss/aux_float"] = avg_aux_float_loss
        
        # Calculate and log functional group metrics
        fg_metrics = compute_metrics(fg_predictions, self.val_dataset.fg_targets)
        
        mlflow_metrics.update({
            "val/fg/accuracy": fg_metrics["overall_accuracy"],
            "val/fg/precision": fg_metrics["overall_precision"],
            "val/fg/recall": fg_metrics["overall_recall"],
            "val/fg/f1": fg_metrics["overall_f1"],
            "val/fg/weighted_f1": fg_metrics["weighted_avg_f1"],
            "val/fg/emr": fg_metrics["exact_match_ratio"],
        })
        
        # Log per-target metrics for functional groups
        for target_idx, target_name in enumerate(self.val_dataset.fg_names):
            mlflow_metrics[f"val/fg/accuracy/{target_name}"] = fg_metrics["per_target_accuracy"][target_idx]
            mlflow_metrics[f"val/fg/precision/{target_name}"] = fg_metrics["per_target_precision"][target_idx]
            mlflow_metrics[f"val/fg/recall/{target_name}"] = fg_metrics["per_target_recall"][target_idx]
            mlflow_metrics[f"val/fg/f1/{target_name}"] = fg_metrics["per_target_f1"][target_idx]
        
        # Calculate and log auxiliary boolean metrics if they exist
        if aux_bool_predictions is not None and hasattr(self.val_dataset, 'aux_bool_targets'):
            aux_bool_metrics = compute_metrics(aux_bool_predictions, self.val_dataset.aux_bool_targets)
            
            mlflow_metrics.update({
                "val/aux_bool/accuracy": aux_bool_metrics["overall_accuracy"],
                "val/aux_bool/precision": aux_bool_metrics["overall_precision"],
                "val/aux_bool/recall": aux_bool_metrics["overall_recall"],
                "val/aux_bool/f1": aux_bool_metrics["overall_f1"],
                "val/aux_bool/weighted_f1": aux_bool_metrics["weighted_avg_f1"],
                "val/aux_bool/emr": aux_bool_metrics["exact_match_ratio"],
            })
            
            # Log per-target metrics for auxiliary boolean targets
            for target_idx, target_name in enumerate(self.val_dataset.aux_bool_names):
                mlflow_metrics[f"val/aux_bool/accuracy/{target_name}"] = aux_bool_metrics["per_target_accuracy"][target_idx]
                mlflow_metrics[f"val/aux_bool/precision/{target_name}"] = aux_bool_metrics["per_target_precision"][target_idx]
                mlflow_metrics[f"val/aux_bool/recall/{target_name}"] = aux_bool_metrics["per_target_recall"][target_idx]
                mlflow_metrics[f"val/aux_bool/f1/{target_name}"] = aux_bool_metrics["per_target_f1"][target_idx]
        
        # Calculate and log auxiliary float metrics if they exist
        if aux_float_predictions is not None and hasattr(self.val_dataset, 'aux_float_targets'):
            # Calculate MSE for each target
            mse_per_target = torch.mean((aux_float_predictions - self.val_dataset.aux_float_targets) ** 2, dim=0)
            
            # Calculate MAE for each target
            mae_per_target = torch.mean(torch.abs(aux_float_predictions - self.val_dataset.aux_float_targets), dim=0)
            
            # Calculate overall MSE and MAE
            overall_mse = torch.mean(mse_per_target)
            overall_mae = torch.mean(mae_per_target)
            
            mlflow_metrics.update({
                "val/aux_float/mse": overall_mse.item(),
                "val/aux_float/mae": overall_mae.item(),
            })
            
            # Log per-target metrics for auxiliary float targets
            for target_idx, target_name in enumerate(self.val_dataset.aux_float_names):
                mlflow_metrics[f"val/aux_float/mse/{target_name}"] = mse_per_target[target_idx].item()
                mlflow_metrics[f"val/aux_float/mae/{target_name}"] = mae_per_target[target_idx].item()
        
        # Log all metrics
        mlflow.log_metrics(mlflow_metrics, step=total_steps)
        
        # Return total loss for early stopping
        return avg_total_loss

    def train(self):
        log_config(self.cfg)
        log_config_params(self.cfg)

        if self.cfg.checkpoint is not None:
            assert is_folder_filename_path(self.cfg.checkpoint), "Checkpoint path should be of form {run_id}/{tag}"
            run_id, tag = self.cfg.checkpoint.split("/")
            self.download_checkpoint(run_id, tag)

        self.best_val_loss = float("inf")
        total_steps = 0  # global step counter
        patience = 0
        max_patience = self.cfg.trainer.patience

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

                    # log batch metrics with separate loss components
                    if (batch_idx + 1) % self.cfg.trainer.log_interval_steps == 0 or batch_idx == len(self.train_loader) - 1:
                        metrics = {
                            "train/loss/total": loss.item(),
                            "train/loss/fg": step_out["fg_loss"].item(),
                            "train/lr": self.optimizer.param_groups[0]["lr"],
                        }
                        
                        # Add auxiliary loss metrics if they exist
                        if "aux_bool_loss" in step_out:
                            metrics["train/loss/aux_bool"] = step_out["aux_bool_loss"].item()
                            
                        if "aux_float_loss" in step_out:
                            metrics["train/loss/aux_float"] = step_out["aux_float_loss"].item()
                            
                        mlflow.log_metrics(metrics, step=total_steps)

                    self.scheduler.step(epoch + batch_idx / len(self.train_loader))  # type: ignore

                    total_steps += 1

                mlflow.log_metric("epoch", epoch + 1, step=total_steps)

                val_loss = self.validate(total_steps=total_steps) 

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best")
                    self.save_checkpoint("latest")
                    patience = 0 
                    mlflow.log_metric("early_stop/patience", patience, step=total_steps)
                else:
                    if val_loss > self.best_val_loss + self.cfg.trainer.patience_threshold:
                        patience += 1
                        mlflow.log_metric("early_stop/patience", patience, step=total_steps)

                        if patience >= max_patience:
                            print(f"Early stopping at epoch {epoch + 1} with patience {patience}.")
                            mlflow.set_tag("early_stop", f"Stopped after {epoch+1} epochs with patience {patience}.")
                            break

                    # if didn't improve, still save every X epochs
                    if (epoch + 1) % self.cfg.trainer.checkpoint_interval_epochs == 0:
                        self.save_checkpoint(f"latest")
        finally:
            upload_sync_artifacts(self.cfg)