from __future__ import annotations

import os
from typing import Callable

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from utils.mlflow_utils import download_artifact

# our transforms are just user defined functions
Transform = Callable[[torch.Tensor], torch.Tensor]

class MLFlowDataset(Dataset):
    """
    Downloads and stores dataset from MLFlow.
    """

    inputs: torch.Tensor # (num_samples, input_features)
    target: torch.Tensor# (num_samples, output_features)

    transform: Transform | None # applied to inputs
    target_names: list[str]
    pos_weights: torch.Tensor # (num_targets,)

    def __init__(self, cfg: DictConfig, dataset_id: str, split: str, transform: Transform | None = None):
        """
        Initialize the dataset.
        Args:
            cfg (DictConfig): Configuration object.
            dataset_id (str): MLFlow run ID of the dataset to download.
            split (str): Split of the dataset to download (train, valid, test).
            transform (Transform | None): Transform to apply to the inputs.
        """
        super().__init__()
        self.cfg = cfg
        self.dataset_id = dataset_id
        self.transform = transform
        self.split = split
        self.download()

    def download(self):
        artifacts = [f"{self.split}_inputs.npy", f"{self.split}_target.npy", "target_names.txt", f"{self.split}_pos_counts.txt"]

        # download only the artifacts we need
        for artifact in artifacts:
            download_artifact(self.cfg, self.dataset_id, artifact)

        local_dir = os.path.join(self.cfg.runs_path, self.dataset_id)

        inputs_path = os.path.join(local_dir, f"{self.split}_inputs.npy")
        target_path = os.path.join(local_dir, f"{self.split}_target.npy")
        target_names_path = os.path.join(local_dir, "target_names.txt")
        pos_counts_path = os.path.join(local_dir, f"{self.split}_pos_counts.txt")

        # verify all exist
        for path in [inputs_path, target_path, target_names_path, pos_counts_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Artifact {path} does not exist")
        
        self.inputs = torch.from_numpy(np.load(inputs_path)).float()
        self.target = torch.from_numpy(np.load(target_path)).float()

        self.target_names = []
        with open(target_names_path, "r") as f:
            for line in f:
                self.target_names.append(line.strip())

        pos_counts = []
        with open(pos_counts_path, "r") as f:
            for line in f:
                pos_counts.append(int(line.strip()))
        
        pos_weights = self._compute_pos_weights(pos_counts)
        self.pos_weights = torch.tensor(pos_weights, dtype=torch.float32)


    def _compute_pos_weights(self, pos_counts: list[int]) -> list[float]:
        """
        Compute positive weights for each target class to address class imbalance.
        
        The positive weight for each class is calculated as (total_samples - pos_count) / pos_count,
        which gives higher weights to underrepresented positive examples.
        
        Args:
            pos_counts (list[int]): Number of positive samples for each target class
            
        Returns:
            list[float]: Positive weights for each target class
        """
        total_samples = self.inputs.shape[0]
        pos_weights = []
        
        for pos_count, target_name in zip(pos_counts, self.target_names):
            # Avoid division by zero
            assert pos_count > 0, f"No positive samples for target {target_name} in the dataset. it probably shouldn't be predicted."
            weight = (total_samples - pos_count) / pos_count
            pos_weights.append(weight)
        
        return pos_weights
    

    def to(self, device: torch.device) -> MLFlowDataset:
        """
        Move the dataset to the specified device.
        :param device: Device to move the dataset to
        """
        self.inputs = self.inputs.to(device)
        self.target = self.target.to(device)
        self.pos_weights = self.pos_weights.to(device)

        return self

    def __len__(self):
        """
        Get the number of samples in the dataset.
        :return: Number of samples
        """
        return self.inputs.shape[0]
    
    def __getitem__(self, index) -> dict:
        """
        Get a sample from the dataset.
        :param index: Index of the sample
        :return: output dictionary
        """

        inputs = self.inputs[index]
        if self.transform is not None:
            inputs = self.transform(inputs)

        target = self.target[index]

        out = {"inputs": inputs, "target": target}

        return out





