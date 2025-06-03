import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random
from typing import Sequence, Literal
import numpy as np

def crop_or_pad(x, max_l, max_d):
    # Get the dimensions of the input tensor x: N, D, H, W
    N, D, H, W = x.shape
    volume_mask = torch.ones(1, D, H, W).long()

    # Handle D dimension (depth)
    if D < max_d:
        # Padding on both sides to make D = L
        pad_D = (max_d - D) // 2
        padding_D = (pad_D, max_d - D - pad_D)  # (left_pad, right_pad)
        x = F.pad(x, (0, 0, 0, 0) + padding_D)  # Pad only D
        volume_mask = F.pad(volume_mask, (0, 0, 0, 0) + padding_D)
    elif D > max_d:
        # If D > L, crop to the center
        start_D = (D - max_d) // 2
        x = x[:, start_D:start_D + max_d, :, :]
        volume_mask = volume_mask[:, start_D:start_D + max_d, :, :]

    # Handle H dimension (height)
    if H < max_l:
        # Padding on both sides to make H = L
        pad_H = (max_l - H) // 2
        padding_H = (pad_H, max_l - H - pad_H)  # (top_pad, bottom_pad)
        x = F.pad(x, (0, 0) + padding_H)  # Pad only H
        volume_mask = F.pad(volume_mask, (0, 0) + padding_H)
    elif H > max_l:
        # If H > L, crop to the center
        start_H = (H - max_l) // 2
        x = x[:, :, start_H:start_H + max_l, :]
        volume_mask = volume_mask[:, :, start_H:start_H + max_l, :]
    
    # Handle W dimension (width)
    if W < max_l:
        # Padding on both sides to make W = L
        pad_W = (max_l - W) // 2
        padding_W = (pad_W, max_l - W - pad_W)  # (left_pad, right_pad)
        x = F.pad(x, padding_W)  # Pad only W
        volume_mask = F.pad(volume_mask, padding_W)
    elif W > max_l:
        # If W > L, crop to the center
        start_W = (W - max_l) // 2
        x = x[:, :, :, start_W:start_W + max_l]
        volume_mask = volume_mask[:, :, :, start_W:start_W + max_l]

    return x, volume_mask


class DepthCrop:
    def __init__(self, img_crop_method: Literal['random', 'center'], d: int):
        assert img_crop_method in ['random', 'center'], "img_crop_method must be 'random' or 'center'"
        self.img_crop_method = img_crop_method
        self.d = d

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (C, D, H, W)
        """
        assert x.dim() == 4, "Input tensor must be 4D (C, D, H, W)"
        C, D, H, W = x.shape
        assert self.d <= D, f"Crop size d={self.d} cannot be larger than D={D}"

        if self.img_crop_method == 'random':
            start = random.randint(0, D - self.d)
        else:  # center
            start = (D - self.d) // 2

        end = start + self.d
        return x[:, start:end, :, :]

class RandomRotation:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


def replace_outliers(image, percentile=0.0001):
    lower_bound, upper_bound = torch.quantile(image, percentile), torch.quantile(
        image, 1 - percentile
    )
    mask = (image <= upper_bound) & (image >= lower_bound)
    valid_pixels = image[mask]
    image[~mask] = torch.clip(image[~mask], min(valid_pixels), max(valid_pixels))

    return image

def normalize(img, global_min, global_max):
    img = np.clip(img, global_min, global_max)
    return (img - global_min) / (global_max - global_min)