from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
import torch.nn.functional as F


class Transform(ABC):
    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass

class Compose(Transform):
    """
    Compose multiple transforms together.

    Args:
        transforms (list): list of Transform objects.
    """
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            x = transform(x)
        return x

    @classmethod
    def from_hydra(cls, cfg: ListConfig) -> Transform:
        """
        Create a Transform object from a Hydra config listing transforms.
        """
        transforms = []
        for transform_cfg in cfg:
            transform = instantiate(transform_cfg)
            transforms.append(transform)

        return cls(transforms)

class AddNoise(Transform):
    def __init__(self, prob: float = 0.2,
                 snr_range: tuple = (2, 10),
                 mean_noise: float = 0.0,
                 eps: float = 1e-12):
        """
        Add Gaussian noise to a 1D (or multi-channel) signal at a random SNR.

        Args:
          prob       float: probability of applying noise
          snr_range  2-tuple: (min_db, max_db) for target SNR in dB; max is exclusive
          mean_noise float: mean of the added noise
          eps        float: small constant inside log10 for stability
        """
        self.prob        = prob
        self.snr_range   = snr_range
        self.mean_noise  = mean_noise
        self.eps         = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: Tensor of shape (C, L) or (L,) with real-valued signal.
        Returns:
          noisy x of same shape.
        """
        if torch.rand((), device=x.device) < self.prob:
            # 1) pick SNR (dB) exactly like np.random.randint
            target_snr_db = int(torch.randint(self.snr_range[0],
                                              self.snr_range[1],
                                              (1,),
                                              device=x.device).item())

            # 2) signal power (Watts), averaged over all elements
            sig_pow = torch.mean(x.pow(2))

            # 3) convert to dB, with eps for numerical stability
            sig_db = 10.0 * torch.log10(sig_pow + self.eps)

            # 4) desired noise power in dB and then Watts
            noise_db  = sig_db - target_snr_db
            noise_pow = torch.pow(10.0, noise_db / 10.0)

            # 5) generate exactly 1-D noise of length L, broadcast across channels
            L = x.shape[-1]
            noise_1d = (torch.randn(L, device=x.device, dtype=x.dtype)
                        * torch.sqrt(noise_pow)
                        + self.mean_noise)
            # if x is multi-channel, broadcast on dim 0
            if x.dim() > 1:
                noise = noise_1d.unsqueeze(0)
            else:
                noise = noise_1d

            x = x + noise

        return x

class Revert(Transform):
    """
    Randomly reverse (time-invert) a 1D signal.

    Args:
        prob (float): probability of applying the reversal.
    """
    def __init__(self, prob: float = 0.2):
        self.prob = prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): 1D signal of shape (L,).

        Returns:
            Tensor: reversed signal with probability `prob`, else unchanged.
        """
        if torch.rand((), device=x.device) < self.prob:
            x = x.flip(dims=[0])
        return x


class MaskZeros(Transform):
    """
    Randomly zero out a fraction of time-steps in a 1D signal.

    Args:
        prob   (float): probability of applying the mask.
        mask_p (tuple): (low, high) fraction of samples to mask.
    """
    def __init__(self, prob: float = 0.2, mask_p: tuple = (0.1, 0.3)):
        self.prob = prob
        self.mask_p = mask_p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): 1D signal of shape (L,).

        Returns:
            Tensor: with `mask_size = uniform(mask_p)*L` random indices set to zero.
        """
        if torch.rand((), device=x.device) < self.prob:
            L = x.shape[0]
            frac = torch.rand((), device=x.device) * (self.mask_p[1] - self.mask_p[0]) + self.mask_p[0]
            mask_size = int((frac * L).item())
            idx = torch.randint(0, L, (mask_size,), device=x.device)
            x = x.clone()
            x[idx] = 0.0
        return x


class ShiftLR(Transform):
    """
    Randomly shift a 1D signal left or right by a fraction of its length.

    Args:
        prob    (float): probability of applying the shift.
        shift_p (tuple): (low, high) fraction of L to shift.
    """
    def __init__(self, prob: float = 0.2, shift_p: tuple = (0.01, 0.05)):
        self.prob = prob
        self.shift_p = shift_p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): 1D signal of shape (L,).

        Returns:
            Tensor: shifted signal (zeros filled) left or right.
        """
        if torch.rand((), device=x.device) < self.prob:
            L = x.shape[0]
            frac = torch.rand((), device=x.device) * (self.shift_p[1] - self.shift_p[0]) + self.shift_p[0]
            shift = int((frac * L).item())

            out = torch.zeros_like(x)
            if torch.rand((), device=x.device) > 0.5:
                # shift right
                out[shift:] = x[:-shift]
            else:
                # shift left
                out[:-shift] = x[shift:]
            x = out
        return x


class ShiftUD(Transform):
    """
    Randomly add or subtract a constant offset (a fraction of max value).

    Args:
        prob    (float): probability of applying the offset.
        shift_p (tuple): (low, high) fraction of max(x) to use as offset.
    """
    def __init__(self, prob: float = 0.2, shift_p: tuple = (0.01, 0.05)):
        self.prob = prob
        self.shift_p = shift_p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): 1D signal of shape (L,).

        Returns:
            Tensor: signal plus or minus offset = max(x) * uniform(shift_p).
        """
        if torch.rand((), device=x.device) < self.prob:
            max_val = x.max()
            frac = torch.rand((), device=x.device) * (self.shift_p[1] - self.shift_p[0]) + self.shift_p[0]
            offset = max_val * frac
            x = x + offset if torch.rand((), device=x.device) > 0.5 else x - offset
        return x


class Normalizer(Transform):
    """
    Min-max normalize a 1D signal to [0,1], optionally standardize.

    Args:
        with_std (bool): whether to subtract mean/divide by std after min-max.
        mean     (float): mean to subtract if `with_std=True`.
        std      (float): std to divide if `with_std=True`.
    """
    def __init__(self,
                 with_std: bool = False,
                 mean: float = 0.0,
                 std:  float = 1.0):
        self.with_std = with_std
        self.mean = mean
        self.std  = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): 1D signal of shape (L,).

        Returns:
            Tensor: normalized (and optionally standardized) signal.
        """
        x = x.to(torch.float32)
        mn, mx = x.min(), x.max()
        norm = (x - mn) / (mx - mn + 1e-12)
        if self.with_std:
            norm = (norm - self.mean) / (self.std + 1e-12)
        return norm

class Resizer:
    """
    Resize a 1D signal to fixed length via bicubic interpolation,
    using PyTorch’s F.interpolate under the hood.
    """

    def __init__(self, signal_size: int = 1024):
        self.signal_size = signal_size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): 1D signal of shape (L,).

        Returns:
            Tensor: resized signal of shape (signal_size,), dtype same as x.
        """
        if x.ndim != 1:
            raise ValueError(f"Expected 1D tensor, got shape {x.shape}")

        # [L] -> [1,1,L,1] so we can use bicubic (2D) interpolation on the length dim
        x4d = x.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # shape (1,1,L,1)

        # interpolate to (signal_size,1)
        y4d = F.interpolate(
            x4d,
            size=(self.signal_size, 1),
            mode="bicubic",
            align_corners=True
        )

        # back to (signal_size,)
        y = y4d.squeeze()  # removes dims 0,1 and the trailing 1

        # make sure dtype and device match the input
        return y.to(dtype=x.dtype, device=x.device)