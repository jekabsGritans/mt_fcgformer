import json
import os
from re import A

import mlflow
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import MLFlowDatasetAggregator
from eval.metrics import compute_metrics
from models import NeuralNetworkModule
from utils.misc import dict_to_device, is_folder_filename_path
from utils.mlflow_utils import (download_artifact, get_run_id, log_config,
                                log_config_params, upload_sync_artifacts)


class Trainer:
    def __init__(self, nn: NeuralNetworkModule, train_agg: MLFlowDatasetAggregator, val_agg: MLFlowDatasetAggregator, cfg: DictConfig):
        self.nn = nn.to(cfg.device)

        self.train_agg = train_agg
        self.val_agg = val_agg

        # Get mask rate from config or use default
        self.train_loader = train_agg.get_loader(
            batch_size=cfg.trainer.batch_size,
            generate_masks=True,
        )
        
        # Validation never uses masks
        self.val_loader = val_agg.get_loader(batch_size=cfg.trainer.batch_size, generate_masks=False)

         
        self.optimizer = torch.optim.AdamW(
            self.nn.parameters(), lr=cfg.trainer.lr, weight_decay=cfg.trainer.weight_decay
        )

        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optimizer, T_0=cfg.trainer.scheduler_t0, T_mult=cfg.trainer.scheduler_tmult)
        # we init scheduler after warmup
        self.scheduler = None
        self.warmup_steps = cfg.trainer.warmup_steps
        self.cosine_first_step = None  # Track when we first apply cosine schedule

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
        if self.cfg.skip_checkpoints:
            return

        run_id = get_run_id()

        local_model_path = os.path.join(self.cfg.runs_path, run_id, f"{tag}_model.pt")
        torch.save(self.nn.state_dict(), local_model_path)

        local_optim_path = os.path.join(self.cfg.runs_path, run_id, f"{tag}_optim.pt")
        torch.save(self.optimizer.state_dict(), local_optim_path)


    def validate(self, total_steps: int, epoch: int) -> float:
        """
        Validate the model on the validation dataset.
        Reports metrics for functional groups and auxiliary targets if they exist.
        :return: Average validation loss
        """
        self.nn.eval()

        # Lists to store predictions and targets
        fg_predictions_list = []
        fg_targets_list = []
        aux_bool_predictions_list = []
        aux_bool_targets_list = []
        aux_float_predictions_list = []
        aux_float_targets_list = []

        total_loss = 0.0
        fg_loss_sum = 0.0
        aux_bool_loss_sum = 0.0
        aux_float_loss_sum = 0.0
        samples_seen = 0

        with torch.no_grad():
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
                
                fg_predictions_list.append(fg_preds)
                fg_targets_list.append(batch["fg_targets"])
                
                # Make predictions for auxiliary boolean targets
                if "aux_bool_logits" in step_out and "aux_bool_targets" in batch:
                    aux_bool_logits = step_out["aux_bool_logits"]
                    aux_bool_preds = torch.sigmoid(aux_bool_logits)
                    aux_bool_preds = (aux_bool_preds > self.cfg.trainer.validator.threshold).float()
                    aux_bool_predictions_list.append(aux_bool_preds)
                    aux_bool_targets_list.append(batch["aux_bool_targets"])
                
                # Make predictions for auxiliary float targets
                if "aux_float_preds" in step_out and "aux_float_targets" in batch:
                    aux_float_predictions_list.append(step_out["aux_float_preds"])
                    aux_float_targets_list.append(batch["aux_float_targets"])
        
        # Concatenate all predictions and targets
        fg_predictions = torch.cat(fg_predictions_list, dim=0)
        fg_targets = torch.cat(fg_targets_list, dim=0)
        
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
        fg_metrics = compute_metrics(fg_predictions, fg_targets)

        overall_val_metrics = {
            "val/fg/accuracy": fg_metrics["overall_accuracy"],
            "val/fg/precision": fg_metrics["overall_precision"],
            "val/fg/recall": fg_metrics["overall_recall"],
            "val/fg/f1": fg_metrics["overall_f1"],
            "val/fg/weighted_f1": fg_metrics["weighted_avg_f1"],
            "val/fg/emr": fg_metrics["exact_match_ratio"],
        }
        
        mlflow_metrics.update(overall_val_metrics)

        # log overall metrics for optuna
        metric_output_file = self.cfg.get("metric_output_file")
        if metric_output_file:
            self.write_metrics_to_file(overall_val_metrics, filepath=metric_output_file)
        
        # Log per-target metrics for functional groups
        for target_idx, target_name in enumerate(self.cfg.fg_names):
            mlflow_metrics[f"val/fg/accuracy/{target_name}"] = fg_metrics["per_target_accuracy"][target_idx]
            mlflow_metrics[f"val/fg/precision/{target_name}"] = fg_metrics["per_target_precision"][target_idx]
            mlflow_metrics[f"val/fg/recall/{target_name}"] = fg_metrics["per_target_recall"][target_idx]
            mlflow_metrics[f"val/fg/f1/{target_name}"] = fg_metrics["per_target_f1"][target_idx]
        
        # Calculate and log auxiliary boolean metrics if they exist
        if aux_bool_predictions_list and len(self.cfg.aux_bool_names) > 0:
            aux_bool_predictions = torch.cat(aux_bool_predictions_list, dim=0)
            aux_bool_targets = torch.cat(aux_bool_targets_list, dim=0)
            aux_bool_metrics = compute_metrics(aux_bool_predictions, aux_bool_targets)
            
            mlflow_metrics.update({
                "val/aux_bool/accuracy": aux_bool_metrics["overall_accuracy"],
                "val/aux_bool/precision": aux_bool_metrics["overall_precision"],
                "val/aux_bool/recall": aux_bool_metrics["overall_recall"],
                "val/aux_bool/f1": aux_bool_metrics["overall_f1"],
                "val/aux_bool/weighted_f1": aux_bool_metrics["weighted_avg_f1"],
                "val/aux_bool/emr": aux_bool_metrics["exact_match_ratio"],
            })
            
            # Log per-target metrics for auxiliary boolean targets
            for target_idx, target_name in enumerate(self.cfg.aux_bool_names):
                mlflow_metrics[f"val/aux_bool/accuracy/{target_name}"] = aux_bool_metrics["per_target_accuracy"][target_idx]
                mlflow_metrics[f"val/aux_bool/precision/{target_name}"] = aux_bool_metrics["per_target_precision"][target_idx]
                mlflow_metrics[f"val/aux_bool/recall/{target_name}"] = aux_bool_metrics["per_target_recall"][target_idx]
                mlflow_metrics[f"val/aux_bool/f1/{target_name}"] = aux_bool_metrics["per_target_f1"][target_idx]
        
        # Calculate and log auxiliary float metrics if they exist
        if aux_float_predictions_list and len(self.cfg.aux_float_names) > 0:
            aux_float_predictions = torch.cat(aux_float_predictions_list, dim=0)
            aux_float_targets = torch.cat(aux_float_targets_list, dim=0)
            
            # Calculate MSE for each target
            mse_per_target = torch.mean((aux_float_predictions - aux_float_targets) ** 2, dim=0)
            
            # Calculate MAE for each target
            mae_per_target = torch.mean(torch.abs(aux_float_predictions - aux_float_targets), dim=0)
            
            # Calculate overall MSE and MAE
            overall_mse = torch.mean(mse_per_target)
            overall_mae = torch.mean(mae_per_target)
            
            mlflow_metrics.update({
                "val/aux_float/mse": overall_mse.item(),
                "val/aux_float/mae": overall_mae.item(),
            })
            
            # Log per-target metrics for auxiliary float targets
            for target_idx, target_name in enumerate(self.cfg.aux_float_names):
                mlflow_metrics[f"val/aux_float/mse/{target_name}"] = mse_per_target[target_idx].item()
                mlflow_metrics[f"val/aux_float/mae/{target_name}"] = mae_per_target[target_idx].item()
        
        # Log all metrics
        mlflow.log_metrics(mlflow_metrics, step=total_steps)
        
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
                self.update_aux_loss_weights(epoch)
                self.nn.train()

                for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch [{epoch+1}/{self.cfg.trainer.epochs}]")):
                    if total_steps < self.cfg.trainer.warmup_steps:
                        # Linear warmup
                        warmup_factor = float(total_steps) / float(max(1, self.cfg.trainer.warmup_steps))
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.cfg.trainer.lr * warmup_factor
                    else:
                        if self.scheduler is None:
                            self.cosine_first_step = total_steps
                            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                                optimizer=self.optimizer,
                                T_0=self.cfg.trainer.scheduler_t0, 
                                T_mult=self.cfg.trainer.scheduler_tmult,
                                eta_min=1e-6  # Add a minimum LR to prevent it from going to zero
                            )

                        cosine_steps = total_steps - self.cosine_first_step # type: ignore
                        cosine_epochs = cosine_steps / len(self.train_loader)
                        self.scheduler.step(cosine_epochs) # type: ignore

                    batch = dict_to_device(batch, self.cfg.device)

                    step_out = self.nn.step(batch)
                    loss = step_out["loss"]

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.nn.parameters(), max_norm=1.0)
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

                    total_steps += 1

                mlflow.log_metric("epoch", epoch + 1, step=total_steps)

                val_loss = self.validate(total_steps=total_steps, epoch=epoch) 

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
    
    def write_metrics_to_file(self, metrics: dict, filepath: str):
        """Write metrics to a JSON file for Optuna to read"""
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Write metrics to file
        with open(filepath, 'w') as f:
            json.dump(metrics, f)

    def update_aux_loss_weights(self, epoch):
        """
        Update auxiliary loss weights with linear decay to zero
        """
        # Get total number of epochs and maximum epoch to decay until
        total_epochs = self.cfg.trainer.epochs
        
        # Start with the initial weights from the config
        initial_aux_bool_weight = self.cfg.trainer.initial_aux_bool_weight
        initial_aux_float_weight = self.cfg.trainer.initial_aux_float_weight
        
        # Calculate linear decay factor (1.0 -> 0.0 over the course of training)
        decay_factor = max(0.0, 1.0 - epoch / self.cfg.trainer.aux_epochs)
        
        # Update the weights
        self.nn.aux_bool_loss_weight = initial_aux_bool_weight * decay_factor
        self.nn.aux_float_loss_weight = initial_aux_float_weight * decay_factor
        
        # Log current weights to MLflow
        mlflow.log_metrics({
            "train/weights/aux_bool": self.nn.aux_bool_loss_weight,
            "train/weights/aux_float": self.nn.aux_float_loss_weight
        }, step=epoch)