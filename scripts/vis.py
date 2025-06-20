#!/usr/bin/env python3
# filepath: /workspace/scripts/find_best_metrics.py
import argparse
import json
import urllib.parse
from pprint import pprint

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient


def configure_mlflow_auth():
    """Configure MLFlow authentication using credentials"""
    MLFLOW_DOMAIN = "mlflow.gritans.lv"
    MLFLOW_USERNAME = "user"
    MLFLOW_PASSWORD = "ko8vohr2EiJunait"
    MLFLOW_TRACKING_URI = f"https://{MLFLOW_DOMAIN}"

    parsed_uri = urllib.parse.urlparse(MLFLOW_TRACKING_URI)
    auth_uri = parsed_uri._replace(
        netloc=f"{urllib.parse.quote(MLFLOW_USERNAME)}:{urllib.parse.quote(MLFLOW_PASSWORD)}@{parsed_uri.netloc}"
    ).geturl()

    mlflow.set_tracking_uri(auth_uri)
    return MlflowClient()

def get_best_metrics(client, run_id):
    """
    Find the step with highest val/fg/weighted_f1 and get all metrics for that step
    
    Args:
        client: MLflow client
        run_id: MLflow run ID
        
    Returns:
        Dictionary containing best metrics
    """
    # Get the metric history for weighted_f1
    # weighted_f1_history = client.get_metric_history(run_id, "val/fg/weighted_f1")
    weighted_f1_history = client.get_metric_history(run_id, "val/fg/f1")
    
    if not weighted_f1_history:
        raise ValueError(f"No 'val/fg/weighted_f1' metric found for run {run_id}")
    
    # Find the step with the highest weighted_f1
    best_step = max(weighted_f1_history, key=lambda x: x.value)
    best_step_number = best_step.step
    best_weighted_f1 = best_step.value
    
    print(f"Best weighted F1 score ({best_weighted_f1:.4f}) found at step {best_step_number}")
    
    # Get all metrics for this run
    run = client.get_run(run_id)
    metrics_at_best_step = {}
    
    # Track all metrics at the best step
    for key in run.data.metrics.keys():
        metric_history = client.get_metric_history(run_id, key)
        for metric in metric_history:
            if metric.step == best_step_number:
                metrics_at_best_step[key] = metric.value
    
    # Extract F1 scores for each functional group
    fg_f1_scores = {}
    for key, value in metrics_at_best_step.items():
        if key.startswith("val/fg/f1/"):
            # Extract functional group name from the key
            fg_name = key.replace("val/fg/f1/", "")
            fg_f1_scores[fg_name] = value
    
    # Get EMR (Exact Match Ratio)
    emr = metrics_at_best_step.get("val/fg/emr", None)
    
    # Create results dictionary
    results = {
        "run_id": run_id,
        "step": best_step_number,
        "weighted_f1": best_weighted_f1,
        "emr": emr,
        "functional_group_f1": fg_f1_scores,
        "epoch": metrics_at_best_step.get("epoch", None)
    }
    
    return results

def visualize_f1_scores(metrics, save_path=None):
    """Create a horizontal bar chart of F1 scores by functional group"""
    # Create DataFrame from functional group F1 scores
    fg_scores = metrics["functional_group_f1"]
    df = pd.DataFrame(list(fg_scores.items()), columns=['Functional Group', 'F1 Score'])
    df = df.sort_values('F1 Score', ascending=False)
    
    # Set up the figure
    plt.figure(figsize=(12, 8))
    
    # Create horizontal bar chart
    bars = plt.barh(df['Functional Group'], df['F1 Score'], color='#4285F4')
    
    # Add value labels
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{df["F1 Score"].iloc[i]:.3f}', 
                va='center', fontsize=10)
    
    # Add weighted F1 reference line
    plt.axvline(metrics["weighted_f1"], color='#DB4437', linestyle='--', linewidth=2, 
                label=f'Weighted F1: {metrics["weighted_f1"]:.3f}')
    
    # Add EMR reference line if available
    if metrics["emr"] is not None:
        plt.axvline(metrics["emr"], color='#0F9D58', linestyle=':', linewidth=2,
                   label=f'Exact Match Ratio: {metrics["emr"]:.3f}')
    
    # Set up chart formatting
    plt.xlabel('F1 Score', fontsize=12)
    plt.title(f'F1 Scores by Functional Group (Run: {metrics["run_id"][:8]}...)', fontsize=14, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.xlim(0, 1.05)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path)
        print(f"Chart saved to {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Find best metrics for an MLflow run")
    parser.add_argument("run_id", help="MLflow run ID")
    parser.add_argument("--save", help="Path to save visualization", default=None)
    parser.add_argument("--export", help="Export results to JSON file", default=None)
    parser.add_argument("--no-viz", action="store_true", help="Don't show visualization")
    args = parser.parse_args()
    
    # Configure MLflow
    client = configure_mlflow_auth()
    
    # Get best metrics
    try:
        metrics = get_best_metrics(client, args.run_id)
        
        # Print results
        print("\nF1 Scores by Functional Group:")
        sorted_fg = sorted(metrics["functional_group_f1"].items(), key=lambda x: x[1], reverse=True)
        for fg_name, f1_score in sorted_fg:
            print(f"  {fg_name}: {f1_score:.4f}")
        
        print(f"\nWeighted F1: {metrics['weighted_f1']:.4f}")
        print(f"Exact Match Ratio: {metrics['emr']:.4f}")
        if metrics['epoch']:
            print(f"Epoch: {metrics['epoch']}")
        
        # Export results if requested
        if args.export:
            with open(args.export, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"\nResults exported to {args.export}")
        
        # Visualize results
        if not args.no_viz:
            visualize_f1_scores(metrics, args.save)
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())