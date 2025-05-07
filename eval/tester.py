import mlflow
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import BaseDataset
from eval.metrics import (compute_exact_match_ratio, compute_overall_accuracy,
                          compute_per_class_accuracy)
from models import BaseModel
from utils.config import get_config
from utils.misc import dict_to_device
from utils.mlflow_utils import download_artifact


class Tester:
    """
    This evaluates the model on the test dataset.
    """
    def __init__(self, model: BaseModel, test_dataset: BaseDataset):
        cfg = get_config()

        self.model = model.to(cfg.device)
        self.dataset = test_dataset.to(cfg.device)

        self.data_loader = DataLoader(
            self.dataset,
            batch_size=cfg.tester.batch_size, shuffle=False, num_workers=cfg.tester.num_workers,
            pin_memory=cfg.tester.pin_memory, persistent_workers=cfg.tester.persistent_workers
        )

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model state from checkpoint.
        Args:
            checkpoint_path (str): Local path to the checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
    
    def download_checkpoint(self, run_id: str, filename: str):
        """
        Download (and load) the checkpoint file from MLFlow.
        Args:
            run_id (str): MLFlow run ID.
            filename (str): Name of the checkpoint file (e.g. "latest_model.pt")
        """
        local_path = download_artifact(run_id=run_id, filename=filename)
        self.load_checkpoint(local_path)

    def test(self):
        """
        Evaluate the model on the test dataset.
        Log results to MLFlow.
        """

        assert self.dataset.target is not None, "Test dataset must have targets for evaluation."

        cfg = get_config()

        predictions = torch.zeros_like(self.dataset.target, device=cfg.device) # (num_samples, num_classes) 0/1 for each class

        self.model.eval()

        with torch.no_grad():
            start_idx = 0
            for batch in tqdm(self.data_loader, desc="Testing", unit="batch"):
                batch = dict_to_device(batch, cfg.device)
                step_out = self.model.step(batch)
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
