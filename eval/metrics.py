"""
Utility functions to compute different metrics and visualizations for validation/testing.
"""
import numpy as np
import torch


def compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict:

    # per-target confusion
    tp = torch.logical_and(preds, targets).sum(dim=0)
    tn = torch.logical_and(torch.logical_not(preds), torch.logical_not(targets)).sum(dim=0)
    fp = torch.logical_and(preds, torch.logical_not(targets)).sum(dim=0)
    fn = torch.logical_and(torch.logical_not(preds), targets).sum(dim=0)

    per_target_acc = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    per_target_precision = tp / (tp + fp + 1e-10)
    per_target_recall = tp / (tp + fn + 1e-10)
    per_target_f1 = 2 * (per_target_precision * per_target_recall) / (per_target_precision + per_target_recall + 1e-10)

    # Calculate support (number of actual occurrences) for each target
    support = targets.sum(dim=0)
    total_support = support.sum()
    
    # Calculate weighted average F1 score
    # Weight each target's F1 score by its proportion in the dataset
    weighted_avg_f1 = torch.sum((support / total_support) * per_target_f1)

    # overall confusion
    overall_tp = tp.sum()
    overall_tn = tn.sum()
    overall_fp = fp.sum()
    overall_fn = fn.sum()

    overall_acc = (overall_tp + overall_tn) / (overall_tp + overall_tn + overall_fp + overall_fn + 1e-10)
    overall_precision = overall_tp / (overall_tp + overall_fp + 1e-10)
    overall_recall = overall_tp / (overall_tp + overall_fn + 1e-10)
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall + 1e-10)

    exact_match_ratio = torch.all(preds == targets, dim=1).float().mean()

    out_lists = {
        "per_target_accuracy": per_target_acc,
        "per_target_precision": per_target_precision,
        "per_target_recall": per_target_recall,
        "per_target_f1": per_target_f1,
        "per_target_support": support,
    }

    out_vals = {
        "overall_accuracy": overall_acc,
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_f1": overall_f1, 
        "weighted_avg_f1": weighted_avg_f1,
        "exact_match_ratio": exact_match_ratio
    }

    out_lists = {k: [round(float(x), 4) for x in v.cpu().numpy()] for k, v in out_lists.items()}   
    out_vals = {k: round(float(v.cpu().numpy()), 4) for k, v in out_vals.items()}

    out = {**out_lists, **out_vals}

    return out