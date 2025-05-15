import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import MLFlowDataset
from eval.metrics import (compute_exact_match_ratio, compute_overall_accuracy,
                          compute_per_class_accuracy)
from models import NeuralNetworkModule
from utils.misc import dict_to_device, is_folder_filename_path
from utils.mlflow_utils import download_artifact, log_config


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

        if self.cfg.checkpoint is not None:
            assert is_folder_filename_path(self.cfg.checkpoint), "Checkpoint path should be of form {run_id}/{tag}"
            run_id, tag = self.cfg.checkpoint.split("/")
            self.download_checkpoint(run_id, tag)

        predictions = torch.zeros_like(self.dataset.target, device=self.cfg.device) # (num_samples, num_classes) 0/1 for each class

        self.nn.eval()

        with torch.no_grad():
            start_idx = 0
            for batch in tqdm(self.data_loader, desc="Testing", unit="batch"):
                batch = dict_to_device(batch, self.cfg.device)
                step_out = self.nn.step(batch)
                logits = step_out["logits"] 

                preds = torch.sigmoid(logits)
                preds = (preds > 0.5).float()

                batch_size = preds.shape[0]
                end_idx = start_idx + batch_size
                predictions[start_idx:end_idx] = preds

                start_idx = end_idx

        per_class_acc = compute_per_class_accuracy(predictions, self.dataset.target)
        overall_acc = compute_overall_accuracy(predictions, self.dataset.target)
        exact_match_ratio = compute_exact_match_ratio(predictions, self.dataset.target)

        test_metrics = {
            "overall_accuracy": overall_acc,
            "per_class_accuracy": per_class_acc,
            "exact_match_ratio": exact_match_ratio,
        }

        print("Test metrics:", test_metrics)

        # TODO: other metrics and visualizations
        # TODO: logg to mlflow. bit different than for training since no epoch
