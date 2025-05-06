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
    transform: Transform | None # applied to inputs
    class_names: list[str] | None 

    def __init__(self, transform: Transform | None, class_names: list[str] | None):
        """
        Initialize the dataset.
        """
        super().__init__()

        self.transform = transform
        self.class_names = class_names

        self.target = None
        self.class_names = None
        self.pos_weights = None

    def get_pos_weights(self) -> torch.Tensor:
        """
        Compute the weights needed for BCE loss to handle class imbalance.
        """
        if self.pos_weights is None:
            assert self.target is not None, "Target labels are not set. Cannot compute pos weights."
            samples_per_class = self.target.sum(dim=0)  # Number of positive samples per class
            total_samples = self.target.shape[0]
            neg_samples = total_samples - samples_per_class
            self.pos_weights = neg_samples / (samples_per_class + 1e-6)  # Avoid div by 0

        return self.pos_weights

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

        if self.transform is not None:
            inputs = self.transform(inputs)

        out = {"inputs": inputs}

        if self.target is not None:
            target = self.target[index]
            out["target"] = target

        return out




