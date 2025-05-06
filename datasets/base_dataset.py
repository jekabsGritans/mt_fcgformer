from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.transforms import np_to_torch

# our transforms are just user defined functions
Transform = Callable[[np.ndarray], torch.Tensor]

class BaseDataset(Dataset):
    """
    Base class for multiclass classification datasets.
    Generates random data for testing.
    """

    inputs: np.ndarray # (num_samples, num_features). out input is 1d
    target: np.ndarray | None # (num_samples, num_classes). one-hot encoded
    transform: Transform
    class_names: list[str] | None
    pos_weights: np.ndarray | None # (num_classes,)

    def __init__(self):
        """
        Initialize the dataset.
        """
        super().__init__()

        # This is fixed per dataset. I.e. we won't want to change with params. 
        # For training augmentations prolly fixed as well, but stochastic and myb dependant on state.
        self.transform = np_to_torch 

        self.target = None
        self.class_names = None
        self.pos_weights = None

        self.load_data()

        if self.target is not None:
            self.compute_pos_weights()
    
    def load_data(self):
        """
        Load the data from the dataset.
        This generates random data for testing.
        :return: Tuple of (inputs, target)
        """
        self.inputs = np.random.rand(100, 10)
        self.target = np.random.randint(0, 2, (100, 5))
    
    def compute_pos_weights(self):
        """
        Compute the positive weights for each class for weighted loss.
        """
        assert self.target is not None, "Target is None. Cannot compute positive weights."

        N = self.target.shape[0]
        samples_per_class = np.sum(self.target, axis=0)
        self.pos_weights = (N - self.target) / (samples_per_class + 1e-6)

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
            target = torch.from_numpy(target).to(torch.float32)
            out["target"] = target

        return out




