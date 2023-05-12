import torch
import numpy as np


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert an image or a batch of images from tensor to ndarray.

    Args:
        tensor (torch.Tensor): The tensor with shape `[c, h, w]` or
        `[b, c, h, w]`.

    Returns:
        np.ndarray: The array with shape `[h, w, c]` or `[b, h, w, c]`.
    """
    if len(tensor.shape) == 3:
        return tensor.detach().permute(1, 2, 0).numpy()
    elif len(tensor.shape) == 4:
        return tensor.detach().permute(0, 2, 3, 1).numpy()


def numpy_to_tensor(array: np.ndarray) -> torch.Tensor:
    """
    Convert batch of images from ndarray to tensor.

    Args:
        tensor (torch.Tensor): The array with shape `[h, w, c]` or
        `[b, h, w, c]`.

    Returns:
        np.ndarray: The tensor with shape `[c, h, w]` or `[b, c, h, w]`.
    """
    if len(array.shape) == 3:
        return torch.tensor(array.transpose(2, 0, 1))
    elif len(array.shape) == 4:
        return torch.tensor(array.transpose(0, 3, 1, 2))
