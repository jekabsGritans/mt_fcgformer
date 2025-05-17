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

        predictions = torch.zeros_like(self.dataset.target, device=self.cfg.device) # (num_samples, num_targets) 0/1 for each target

        self.nn.eval()

        with torch.no_grad():
            start_idx = 0
            for batch in tqdm(self.data_loader, desc="Testing", unit="batch"):
                batch = dict_to_device(batch, self.cfg.device)
                step_out = self.nn.step(batch)
                logits = step_out["logits"] 

                preds = torch.sigmoid(logits)
                preds = (preds > self.cfg.tester.threshold).float()

                batch_size = preds.shape[0]
                end_idx = start_idx + batch_size
                predictions[start_idx:end_idx] = preds

                start_idx = end_idx

        metrics = compute_metrics(predictions, self.dataset.target)

        mlflow_metrics = {
            "overall/accuracy": metrics["overall_accuracy"],
            "overall/precision": metrics["overall_precision"],
            "overall/recall": metrics["overall_recall"],
            "overall/f1": metrics["overall_f1"],
            "overall/emr": metrics["exact_match_ratio"]
        }

        for target_idx, target_name in enumerate(self.dataset.target_names):
            mlflow_metrics[f"per_target/accuracy/{target_name}"] = metrics["per_target_accuracy"][target_idx]
            mlflow_metrics[f"per_target/precision/{target_name}"] = metrics["per_target_precision"][target_idx]
            mlflow_metrics[f"per_target/recall/{target_name}"] = metrics["per_target_recall"][target_idx]
            mlflow_metrics[f"per_target/f1/{target_name}"] = metrics["per_target_f1"][target_idx]

        mlflow.log_metrics(mlflow_metrics)

        # bar plot for per-target metrics
        fig, ax = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Per-Target Metrics", fontsize=18, y=0.98)
        
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
                range(len(self.dataset.target_names)), 
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
            ax[row, col].set_xticks(range(len(self.dataset.target_names)))
            ax[row, col].set_xticklabels(self.dataset.target_names, rotation=45, ha='right', fontsize=10)
            
            # Add grid for better readability
            ax[row, col].grid(axis='y', linestyle='--', alpha=0.7)
            ax[row, col].set_axisbelow(True)
            
            # Set y-axis to start at 0 and have a reasonable upper limit
            ax[row, col].set_ylim(0, min(1.1, max(metrics_data[i]) * 1.2))
        
        # Adjust layout to make room for rotated labels and avoid overlap
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        plt.subplots_adjust(left=0.08, right=0.92, top=0.9, bottom=0.15)
        
        # Save the figure with tight layout
        mlflow.log_figure(fig, "per_target_metrics.png")
        plt.close(fig)

