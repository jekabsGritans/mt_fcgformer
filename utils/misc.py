import torch


def dict_to_device(d: dict, device: str) -> dict:
    """
    Move tensors in a dictionary to a specific device.
    :param d: Dictionary of tensors
    :param device: Device to move the tensors to
    :return: Dictionary of tensors on the specified device
    """
    return {k: v.to(device) for k, v in d.items() if isinstance(v, torch.Tensor)}