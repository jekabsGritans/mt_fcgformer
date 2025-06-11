from pathlib import Path

import numpy as np
import torch


def dict_to_device(d: dict, device: str) -> dict:
    """
    Move tensors in a dictionary to a specific device.
    :param d: Dictionary of tensors
    :param device: Device to move the tensors to
    :return: Dictionary of tensors on the specified device
    """
    return {k: v.to(device) for k, v in d.items() if isinstance(v, torch.Tensor)}



def is_folder_filename_path(path_str: str) -> bool:
    """
    Check if path is of form {folder}/{filename}
    """
    path = Path(path_str)
    return (
        len(path.parts) == 2 and      # exactly one folder + one file
        path.parent != Path(".") and  # has a parent folder
        not path_str.endswith("/")    # does not end with slash (not a directory)
    )

def interpolate(x: np.ndarray, y: np.ndarray, min_x: float, max_x: float, num_points: int) -> np.ndarray:
    """
    Interpolate y values for a given x range.
    
    :param x: Original x values
    :param y: Original y values
    :param min_x: Minimum x value for interpolation
    :param max_x: Maximum x value for interpolation
    :param num_points: Number of points in the interpolated range
    :return: Interpolated y values
    """
    new_x = np.linspace(min_x, max_x, num_points)
    return np.interp(new_x, x, y)