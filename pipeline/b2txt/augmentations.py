from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d


def gauss_smooth(
    inputs: torch.Tensor,
    device: torch.device,
    smooth_kernel_std: float = 2.0,
    smooth_kernel_size: int = 100,
    padding: str = "same",
) -> torch.Tensor:
    """Apply 1D Gaussian smoothing along the time axis."""
    kernel = np.zeros(smooth_kernel_size, dtype=np.float32)
    kernel[smooth_kernel_size // 2] = 1
    kernel = gaussian_filter1d(kernel, smooth_kernel_std)
    valid_idx = np.argwhere(kernel > 0.01)
    kernel = kernel[valid_idx]
    kernel = np.squeeze(kernel / np.sum(kernel))

    kernel_tensor = torch.tensor(kernel, dtype=torch.float32, device=device)
    kernel_tensor = kernel_tensor.view(1, 1, -1)

    batch, time_steps, channels = inputs.shape
    inputs = inputs.permute(0, 2, 1)
    kernel_tensor = kernel_tensor.repeat(channels, 1, 1)

    smoothed = F.conv1d(inputs, kernel_tensor, padding=padding, groups=channels)
    return smoothed.permute(0, 2, 1)


def apply_transforms(
    features: torch.Tensor,
    n_time_steps: torch.Tensor,
    transforms: Dict[str, object],
    device: torch.device,
    mode: str = "train",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply data augmentations and smoothing."""
    if not transforms:
        return features, n_time_steps

    data_shape = features.shape
    batch_size = data_shape[0]
    channels = data_shape[-1]

    if mode == "train":
        static_gain_std = transforms.get("static_gain_std", 0.0)
        if static_gain_std and static_gain_std > 0:
            warp_mat = torch.eye(channels, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
            warp_mat += torch.randn_like(warp_mat) * static_gain_std
            features = torch.matmul(features, warp_mat)

        white_noise_std = transforms.get("white_noise_std", 0.0)
        if white_noise_std and white_noise_std > 0:
            features = features + torch.randn(data_shape, device=device) * white_noise_std

        constant_offset_std = transforms.get("constant_offset_std", 0.0)
        if constant_offset_std and constant_offset_std > 0:
            features = features + torch.randn((batch_size, 1, channels), device=device) * constant_offset_std

        random_walk_std = transforms.get("random_walk_std", 0.0)
        if random_walk_std and random_walk_std > 0:
            axis = transforms.get("random_walk_axis", -1)
            features = features + torch.cumsum(
                torch.randn(data_shape, device=device) * random_walk_std,
                dim=axis,
            )

        random_cut = transforms.get("random_cut", 0)
        if random_cut and random_cut > 0:
            cut = int(np.random.randint(0, random_cut))
            if cut > 0:
                features = features[:, cut:, :]
                n_time_steps = n_time_steps - cut

    if transforms.get("smooth_data", False):
        features = gauss_smooth(
            inputs=features,
            device=device,
            smooth_kernel_std=transforms.get("smooth_kernel_std", 2.0),
            smooth_kernel_size=transforms.get("smooth_kernel_size", 100),
        )

    return features, n_time_steps
