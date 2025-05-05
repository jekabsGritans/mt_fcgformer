import numpy as np
import torch


def np_to_torch(x: np.ndarray) -> torch.Tensor:
    """
    Convert a numpy array to a torch tensor.
    """
    return torch.from_numpy(x).to(torch.float32)