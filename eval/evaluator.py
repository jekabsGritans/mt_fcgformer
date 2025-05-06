import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import BaseDataset
from models import BaseModel
from utils.misc import dict_to_device
from utils.mlflow_utils import MLFlowManager


class Evaluator:
    """
    This evaluates the model on a dataset.
    Used for validation during training and for final evaluation during testing.
    """
    def __init__(self, model: BaseModel, eval_dataset: BaseDataset, device: str, batch_size: int, num_workers: int, pin_memory: bool, persistent_workers: bool, mlflow_manager: MLFlowManager):
        self.model = model
        self.eval_dataset = eval_dataset
        self.device = device
        self.batch_size = batch_size
        self.mlflow_manager = mlflow_manager

        self.eval_loader = DataLoader(
            self.eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers
            )
        
        self.model.to(self.device)

    def evaluate(self, step: int = 0) -> tuple[float, float]:
        """
        Evaluate the model on the validation dataset.
        :return: Tuple of (average loss, average accuracy)
        """

        self.model.eval()
        total_loss = 0
        total_acc = 0
        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating", unit="batch"):
                batch = dict_to_device(batch, self.device)
                step_out = self.model.step(batch)
                logits = step_out["logits"] 
                loss = step_out["loss"]
                total_loss += loss.item()

                preds = torch.sigmoid(logits)
                preds = (preds > 0.5).float()
                targets = batch["target"]
                acc = (preds == targets).float().mean().item()
                total_acc += acc

        avg_loss = total_loss / len(self.eval_loader)
        avg_acc = total_acc / len(self.eval_loader)

        metrics = {
            "loss": avg_loss,
            "accuracy": avg_acc,
        }

        self.mlflow_manager.log_metrics(metrics, step=step)
        print(f"[Evaluation] Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")

        return avg_loss, avg_acc