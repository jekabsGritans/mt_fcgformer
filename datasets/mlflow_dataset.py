from __future__ import annotations

import os
from abc import ABC
from typing import Callable

import mlflow
import mlflow.artifacts
import numpy as np
import torch
from torch.utils.data import Dataset

# our transforms are just user defined functions
Transform = Callable[[torch.Tensor], torch.Tensor]

class MLFlowDataset(Dataset):
    """
    Downloads and stores dataset from MLFlow.
    """

    inputs: torch.Tensor # (num_samples, input_features)
    target: torch.Tensor# (num_samples, output_features)

    transform: Transform | None # applied to inputs
    class_names: list[str]

    def __init__(self, dataset_id: str, transform: Transform | None = None):
        """
        Initialize the dataset.
        Args:
            dataset_id (str): MLFlow run ID of the dataset to download.
            transform (Transform | None): Transform to apply to the inputs.
        """
        super().__init__()
        self.dataset_id = dataset_id
        self.transform = transform
        self.download()

    def download(self):
        local_dir = mlflow.artifacts.download_artifacts(run_id=self.dataset_id)
        inputs_path = os.path.join(local_dir, "inputs.npy")
        target_path = os.path.join(local_dir, "target.npy")
        target_names_path = os.path.join(local_dir, "target_names.txt")
        pos_counts_path = os.path.join(local_dir, "pos_counts.txt")

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
        raise NotImplementedError()
    

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





