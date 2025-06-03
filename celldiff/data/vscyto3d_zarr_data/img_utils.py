import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random
from typing import Sequence, Tuple
import numpy as np
from typing import Literal

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


class RandomRotation:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


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

class RandomCropResize:
    def __init__(self, spatial_size: int, rescale_range: Tuple[float, float]):
        self.spatial_size = spatial_size
        self.rescale_range = rescale_range

    def __call__(self, img: torch.Tensor):
        h, w = img.shape[-2], img.shape[-1]
        scale_factor = random.uniform(*self.rescale_range)
        crop_size = int(self.spatial_size * scale_factor)

        # Ensure crop_size doesn't exceed image size
        crop_size = min(crop_size, h, w)

        top = random.randint(0, h - crop_size)
        left = random.randint(0, w - crop_size)
        img = TF.crop(img, top, left, crop_size, crop_size)

        # Resize to target size
        img = TF.resize(img, self.spatial_size)

        return img

class RandomAdjustContrast:
    def __init__(self, contrast_range: Tuple[float, float]):
        self.contrast_range = contrast_range

    def __call__(self, img: torch.Tensor):
        factor = random.uniform(*self.contrast_range)

        mean = img.mean(dim=(-3, -2, -1), keepdim=True)
        out = (img - mean) * factor + mean

        return out.clamp(0, 1)

class RandomAdjustGamma:
    def __init__(self, gamma_range: Tuple[float, float], eps: float = 1e-6):
        self.gamma_range = gamma_range
        self.eps = eps

    def __call__(self, img: torch.Tensor):
        gamma = random.uniform(*self.gamma_range)
        return img.clamp(min=self.eps).pow(gamma).clamp(0, 1)

class RandomGaussianBlur:
    def __init__(self, sigma_range: Tuple[float, float]):
        self.sigma_range = sigma_range

    def __call__(self, img: torch.Tensor):
        """
        Args:
            img: (C, H, W) float tensor in [0, 1]
        """
        sigma = random.uniform(*self.sigma_range)
        return TF.gaussian_blur(img, kernel_size=self._get_kernel_size(sigma), sigma=sigma)

    def _get_kernel_size(self, sigma: float):
        # Kernel size should be odd and sufficiently large to cover the Gaussian
        kernel_size = int(2 * round(3 * sigma) + 1)
        return kernel_size if kernel_size % 2 == 1 else kernel_size + 1

class RandomGaussianNoise:
    def __init__(self, mean_range: Tuple[float, float], std_range: Tuple[float, float]):
        self.mean_range = mean_range
        self.std_range = std_range

    def __call__(self, img: torch.Tensor):
        mean = random.uniform(*self.mean_range)
        std = random.uniform(*self.std_range)

        noise = torch.randn_like(img) * std + mean
        return (img + noise).clamp(0, 1)


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