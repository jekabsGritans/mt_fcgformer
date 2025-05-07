"""
Utility functions to compute different metrics and visualizations for validation/testing.
"""
import torch


def compute_overall_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute overall accuracy of the model predictions.
    
    Args:
        preds (torch.Tensor): (num_samples, num_classes) predictions (0/1 for each class).
        targets (torch.Tensor): (num_samples, num_classes) ground truth labels (0/1 for each class).
    """

    assert preds.shape == targets.shape, "Predictions and targets must have the same shape."
    
    # Compute overall accuracy
    correct = (preds == targets).float().sum()
    total = targets.numel()
    accuracy = correct / total
    
    return accuracy.item()

def compute_per_class_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> dict[int, float]:
    """
    Compute per-class accuracy of the model predictions.
    Args:
        preds (torch.Tensor): (num_samples, num_classes) predictions (0/1 for each class).
        targets (torch.Tensor): (num_samples, num_classes) ground truth labels (0/1 for each class).
    """
    assert preds.shape == targets.shape, "Predictions and targets must have the same shape."
    
    # Compute per-class accuracy
    per_class_acc = {
        i: (preds[:, i] == targets[:, i]).float().mean().item()
        for i in range(preds.shape[1])
    }
    
    return per_class_acc

def compute_exact_match_ratio(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute exact match ratio of the model predictions.
    
    Args:
        preds (torch.Tensor): (num_samples, num_classes) predictions (0/1 for each class).
        targets (torch.Tensor): (num_samples, num_classes) ground truth labels (0/1 for each class).
    """
    assert preds.shape == targets.shape, "Predictions and targets must have the same shape."
    
    # Compute exact match ratio
    exact_match = torch.all(preds == targets, dim=1)
    exact_match_ratio = exact_match.float().mean()
    
    return exact_match_ratio.item()