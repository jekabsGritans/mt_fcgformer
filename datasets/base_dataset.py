from __future__ import annotations

from abc import ABC
from typing import Callable

import torch
from torch.utils.data import Dataset

# our transforms are just user defined functions
Transform = Callable[[torch.Tensor], torch.Tensor]

class BaseDataset(Dataset, ABC):
    """
    Base class for multiclass classification datasets.
    Generates random data for testing.
    """

    inputs: torch.Tensor # (num_samples, num_features)
    target: torch.Tensor | None # (num_samples, num_classes). one-hot encoded

    transform: Transform # applied to inputs
    class_names: list[str]

    def __init__(self, transform: Transform, class_names: list[str], pos_weights: list[float]):
        """
        Initialize the dataset.
        """
        super().__init__()

        self.transform = transform
        self.class_names = class_names
        self.pos_weights = torch.tensor(pos_weights, dtype=torch.float32) 
    
    def to(self, device: torch.device) -> BaseDataset:
        """
        Move the dataset to the specified device.
        :param device: Device to move the dataset to
        """
        self.inputs.to(device)
        self.pos_weights.to(device)

        if self.target is not None:
            self.target.to(device)

        return self

    def get_class_name(self, class_idx: int) -> str:
        """
        Get the name of a predicted class by index.
        """
        if self.class_names is None:
            return f"Class {class_idx}"
        else:
            return self.class_names[class_idx]
       
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
        inputs = self.transform(inputs)

        out = {"inputs": inputs}

        if self.target is not None:
            target = self.target[index]
            out["target"] = target

        return out




