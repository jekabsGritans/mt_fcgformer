import matplotlib.pyplot as plt
import mlflow
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import MLFlowDataset
from eval.metrics import compute_metrics
from models import NeuralNetworkModule
from utils.misc import dict_to_device, is_folder_filename_path
from utils.mlflow_utils import download_artifact, log_config, log_config_params


class Tester:
    """
    This evaluates the model on the test dataset.
    """
    def __init__(self, nn: NeuralNetworkModule, test_dataset: MLFlowDataset, cfg: DictConfig):
        self.nn = nn.to(cfg.device)
        self.dataset = test_dataset.to(cfg.device)

        self.data_loader = DataLoader(
            self.dataset,
            batch_size=cfg.tester.batch_size, shuffle=False, num_workers=cfg.tester.num_workers,
            pin_memory=cfg.tester.pin_memory, persistent_workers=cfg.tester.persistent_workers
        )

        self.cfg = cfg

    def load_checkpoint(self, model_path: str):
        """
        Load model state from checkpoint.
        """
        self.nn.load_state_dict(torch.load(model_path))
    
    def download_checkpoint(self, run_id: str, tag: str):
        """
        Download (and load) the checkpoint file from MLFlow.
        """
        model_path = download_artifact(self.cfg, run_id, f"{tag}_model.pt")
        self.load_checkpoint(model_path)
    
    def test(self):
        """
        Evaluate the model on the test dataset.
        Log results to MLFlow.
        """
        log_config(self.cfg)
        log_config_params(self.cfg)

        if self.cfg.checkpoint is not None:
            assert is_folder_filename_path(self.cfg.checkpoint), "Checkpoint path should be of form {run_id}/{tag}"
            run_id, tag = self.cfg.checkpoint.split("/")
            self.download_checkpoint(run_id, tag)

        # Initialize predictions tensors
        fg_predictions = torch.zeros_like(self.dataset.fg_targets, device=self.cfg.device)
        
        # Initialize aux prediction tensors only if they exist
        if hasattr(self.dataset, 'aux_bool_targets') and self.dataset.aux_bool_targets.shape[1] > 0:
            aux_bool_predictions = torch.zeros_like(self.dataset.aux_bool_targets, device=self.cfg.device)
        else:
            aux_bool_predictions = None
            
        if hasattr(self.dataset, 'aux_float_targets') and self.dataset.aux_float_targets.shape[1] > 0:
            aux_float_predictions = torch.zeros_like(self.dataset.aux_float_targets, device=self.cfg.device)
        else:
            aux_float_predictions = None

        # Track loss components
        total_loss = 0.0
        fg_loss_sum = 0.0
        aux_bool_loss_sum = 0.0
        aux_float_loss_sum = 0.0
        samples_seen = 0

        self.nn.eval()

        with torch.no_grad():
            start_idx = 0
            for batch in tqdm(self.data_loader, desc="Testing", unit="batch"):
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
                fg_preds = (fg_preds > self.cfg.tester.threshold).float()
                
                end_idx = start_idx + batch_size
                fg_predictions[start_idx:end_idx] = fg_preds
                
                # Make predictions for auxiliary boolean targets
                if "aux_bool_logits" in step_out and aux_bool_predictions is not None:
                    aux_bool_logits = step_out["aux_bool_logits"]
                    aux_bool_preds = torch.sigmoid(aux_bool_logits)
                    aux_bool_preds = (aux_bool_preds > self.cfg.tester.threshold).float()
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
            "test/loss/total": avg_total_loss,
            "test/loss/fg": avg_fg_loss
        }
        
        # Add auxiliary losses if they exist
        if aux_bool_loss_sum > 0:
            mlflow_metrics["test/loss/aux_bool"] = avg_aux_bool_loss
            
        if aux_float_loss_sum > 0:
            mlflow_metrics["test/loss/aux_float"] = avg_aux_float_loss

        # Calculate and log functional group metrics
        fg_metrics = compute_metrics(fg_predictions, self.dataset.fg_targets)
        
        mlflow_metrics.update({
            "test/fg/accuracy": fg_metrics["overall_accuracy"],
            "test/fg/precision": fg_metrics["overall_precision"],
            "test/fg/recall": fg_metrics["overall_recall"],
            "test/fg/f1": fg_metrics["overall_f1"],
            "test/fg/weighted_f1": fg_metrics["weighted_avg_f1"],
            "test/fg/emr": fg_metrics["exact_match_ratio"],
        })
        
        # Log per-target metrics for functional groups
        for target_idx, target_name in enumerate(self.dataset.fg_names):
            mlflow_metrics[f"test/fg/accuracy/{target_name}"] = fg_metrics["per_target_accuracy"][target_idx]
            mlflow_metrics[f"test/fg/precision/{target_name}"] = fg_metrics["per_target_precision"][target_idx]
            mlflow_metrics[f"test/fg/recall/{target_name}"] = fg_metrics["per_target_recall"][target_idx]
            mlflow_metrics[f"test/fg/f1/{target_name}"] = fg_metrics["per_target_f1"][target_idx]

        # Calculate and log auxiliary boolean metrics if they exist
        if aux_bool_predictions is not None and hasattr(self.dataset, 'aux_bool_targets'):
            aux_bool_metrics = compute_metrics(aux_bool_predictions, self.dataset.aux_bool_targets)
            
            mlflow_metrics.update({
                "test/aux_bool/accuracy": aux_bool_metrics["overall_accuracy"],
                "test/aux_bool/precision": aux_bool_metrics["overall_precision"],
                "test/aux_bool/recall": aux_bool_metrics["overall_recall"],
                "test/aux_bool/f1": aux_bool_metrics["overall_f1"],
                "test/aux_bool/weighted_f1": aux_bool_metrics["weighted_avg_f1"],
                "test/aux_bool/emr": aux_bool_metrics["exact_match_ratio"],
            })
            
            # Log per-target metrics for auxiliary boolean targets
            for target_idx, target_name in enumerate(self.dataset.aux_bool_names):
                mlflow_metrics[f"test/aux_bool/accuracy/{target_name}"] = aux_bool_metrics["per_target_accuracy"][target_idx]
                mlflow_metrics[f"test/aux_bool/precision/{target_name}"] = aux_bool_metrics["per_target_precision"][target_idx]
                mlflow_metrics[f"test/aux_bool/recall/{target_name}"] = aux_bool_metrics["per_target_recall"][target_idx]
                mlflow_metrics[f"test/aux_bool/f1/{target_name}"] = aux_bool_metrics["per_target_f1"][target_idx]
        
        # Calculate and log auxiliary float metrics if they exist
        if aux_float_predictions is not None and hasattr(self.dataset, 'aux_float_targets'):
            # Calculate MSE for each target
            mse_per_target = torch.mean((aux_float_predictions - self.dataset.aux_float_targets) ** 2, dim=0)
            
            # Calculate MAE for each target
            mae_per_target = torch.mean(torch.abs(aux_float_predictions - self.dataset.aux_float_targets), dim=0)
            
            # Calculate overall MSE and MAE
            overall_mse = torch.mean(mse_per_target)
            overall_mae = torch.mean(mae_per_target)
            
            mlflow_metrics.update({
                "test/aux_float/mse": overall_mse.item(),
                "test/aux_float/mae": overall_mae.item(),
            })
            
            # Log per-target metrics for auxiliary float targets
            for target_idx, target_name in enumerate(self.dataset.aux_float_names):
                mlflow_metrics[f"test/aux_float/mse/{target_name}"] = mse_per_target[target_idx].item()
                mlflow_metrics[f"test/aux_float/mae/{target_name}"] = mae_per_target[target_idx].item()

        mlflow.log_metrics(mlflow_metrics)

        # Create visualizations for functional group metrics
        self._create_target_plots(fg_metrics, self.dataset.fg_names, "fg")
        
        # Create visualizations for auxiliary boolean metrics if they exist
        if aux_bool_predictions is not None and hasattr(self.dataset, 'aux_bool_targets'):
            self._create_target_plots(aux_bool_metrics, self.dataset.aux_bool_names, "aux_bool")
            
        # Create visualizations for auxiliary float metrics if they exist
        if aux_float_predictions is not None and hasattr(self.dataset, 'aux_float_targets'):
            self._create_float_plots(mse_per_target, mae_per_target, self.dataset.aux_float_names)

    def _create_target_plots(self, metrics, target_names, target_type):
        """
        Create and save plots for target metrics.
        """
        fig, ax = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"{target_type.upper()} Target Metrics", fontsize=18, y=0.98)
        
        # Set consistent colors for the plots - using a colorblind-friendly palette
        colors = ['#4e79a7', '#f28e2c', '#59a14f', '#e15759']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        metrics_data = [
            metrics["per_target_accuracy"],
            metrics["per_target_precision"], 
            metrics["per_target_recall"],
            metrics["per_target_f1"]
        ]
        
        # Loop through subplots for consistent formatting
        for i, (row, col) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
            # Create the bar chart
            bars = ax[row, col].bar(
                range(len(target_names)), 
                metrics_data[i],
                color=colors[i],
                width=0.7
            )
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax[row, col].annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8)
            
            # Set titles and labels
            ax[row, col].set_title(f"{metric_names[i]}", fontsize=14)
            ax[row, col].set_ylabel(metric_names[i], fontsize=12)
            
            # Use integer positions for x-ticks but show target names as labels
            ax[row, col].set_xticks(range(len(target_names)))
            ax[row, col].set_xticklabels(target_names, rotation=45, ha='right', fontsize=10)
            
            # Add grid for better readability
            ax[row, col].grid(axis='y', linestyle='--', alpha=0.7)
            ax[row, col].set_axisbelow(True)
            
            # Set y-axis to start at 0 and have a reasonable upper limit
            ax[row, col].set_ylim(0, min(1.1, max(metrics_data[i]) * 1.2))
        
        # Adjust layout to make room for rotated labels and avoid overlap
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        plt.subplots_adjust(left=0.08, right=0.92, top=0.9, bottom=0.15)
        
        # Save the figure with tight layout
        mlflow.log_figure(fig, f"{target_type}_target_metrics.png")
        plt.close(fig)
        
    def _create_float_plots(self, mse_per_target, mae_per_target, target_names):
        """
        Create and save plots for auxiliary float metrics.
        """
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Auxiliary Float Target Metrics", fontsize=18, y=0.98)
        
        # Set consistent colors for the plots - using a colorblind-friendly palette
        colors = ['#4e79a7', '#f28e2c']
        metric_names = ['MSE', 'MAE']
        metrics_data = [mse_per_target, mae_per_target]
        
        # Loop through subplots for consistent formatting
        for i, col in enumerate([0, 1]):
            # Create the bar chart
            bars = ax[col].bar(
                range(len(target_names)), 
                metrics_data[i].cpu(),  # Move to CPU for plotting
                color=colors[i],
                width=0.7
            )
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax[col].annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8)
            
            # Set titles and labels
            ax[col].set_title(f"{metric_names[i]}", fontsize=14)
            ax[col].set_ylabel(metric_names[i], fontsize=12)
            
            # Use integer positions for x-ticks but show target names as labels
            ax[col].set_xticks(range(len(target_names)))
            ax[col].set_xticklabels(target_names, rotation=45, ha='right', fontsize=10)
            
            # Add grid for better readability
            ax[col].grid(axis='y', linestyle='--', alpha=0.7)
            ax[col].set_axisbelow(True)
            
            # Set y-axis to start at 0 and have a reasonable upper limit
            ax[col].set_ylim(0, max(metrics_data[i].cpu().numpy()) * 1.2)
        
        # Adjust layout to make room for rotated labels and avoid overlap
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        plt.subplots_adjust(left=0.08, right=0.92, top=0.9, bottom=0.15)
        
        # Save the figure with tight layout
        mlflow.log_figure(fig, "aux_float_target_metrics.png")
        plt.close(fig)