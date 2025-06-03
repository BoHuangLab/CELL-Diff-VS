# -*- coding: utf-8 -*-
import numpy as np
import torch

from typing import List
from torch.utils.data import Dataset

from .collater import collate_fn

from torchvision import transforms
from .img_utils import (
    RandomRotation, 
    normalize, 
    DepthCrop, 
    RandomCropResize, 
    RandomAdjustContrast, 
    RandomAdjustGamma, 
    RandomGaussianBlur, 
    RandomGaussianNoise, 
)
import zarr


class VSCyto3DZARRDataset(Dataset):
    def __init__(self, args, split_key) -> None:
        super().__init__()
        self.args = args

        self.split_key = split_key
        self.data_path = self.args.data_path

        self.data = zarr.open(self.data_path, mode='r')

        fovs = [fov for fov in self.data[0].keys()]

        val_fovs = ['172', '286', '307', '323', '331']
        test_fovs = ['261', '271', '285', '55', '66']

        train_fovs = [fov for fov in fovs if fov not in val_fovs + test_fovs]

        if split_key == 'train':
            self.fovs = train_fovs
        elif split_key == 'val':
            self.fovs = val_fovs
        elif split_key == 'test':
            self.fovs = test_fovs

        self.phase_global_min = -0.06567
        self.phase_global_max = 0.07901

        self.nucleus_global_min = 0
        self.nucleus_global_max = 11520.41875
        
        self.membrane_global_min = 0
        self.membrane_global_max = 528521.60937

        self.spatial_size = self.args.input_spatial_size
        self.data_aug = self.args.data_aug

    def __getitem__(self, index: int) -> dict:
        fov = self.fovs[index]
        data = self.data[0][fov][0][0][0, 3:]
        
        item = {}
        item['phase'], item['nucleus'], item['membrane'] = self.get_img(data)
        
        return item

    def get_img(self, data):
        img = data.astype(np.float32)

        phase = img[0]
        nucleus = img[1]
        membrane = img[2]

        phase = normalize(phase, self.phase_global_min, self.phase_global_max)
        nucleus = normalize(nucleus, self.nucleus_global_min, self.nucleus_global_max)
        membrane = normalize(membrane, self.membrane_global_min, self.membrane_global_max)

        phase = torch.from_numpy(phase)
        nucleus = torch.from_numpy(nucleus)
        membrane = torch.from_numpy(membrane)

        t_forms = []
        if self.data_aug:
            t_forms.append(RandomCropResize(self.spatial_size[-1], rescale_range=(0.5, 2)))
            t_forms.append(transforms.RandomHorizontalFlip(p=0.5))
            t_forms.append(RandomRotation([0, 90, 180, 270]))
        else:
            t_forms.append(transforms.CenterCrop(self.spatial_size[-1]))
        
        img_crop_method = 'random' if self.data_aug else 'center'
        t_forms.append(DepthCrop(img_crop_method, self.spatial_size[0]))
        t_forms = transforms.Compose(t_forms)

        img = torch.stack([phase, nucleus, membrane], dim=0)
        phase, nucleus, membrane = t_forms(img)

        if self.data_aug:
            phase_t_forms = []
            phase_t_forms.append(RandomAdjustContrast(contrast_range=(0.5, 1.5)))
            phase_t_forms.append(RandomAdjustGamma(gamma_range=(0.5, 1.5)))
            phase_t_forms.append(RandomGaussianNoise(mean_range=(-0.3, 0.3), std_range=(0.0, 0.1)))
            phase_t_forms = transforms.Compose(phase_t_forms)
            phase = phase_t_forms(phase)

        phase = phase.clip(0, 1)
        nucleus = nucleus.clip(0, 1)
        membrane = membrane.clip(0, 1)

        phase = phase * 2 - 1
        nucleus = nucleus * 2 - 1
        membrane = membrane * 2 - 1

        return phase.unsqueeze(0), nucleus.unsqueeze(0), membrane.unsqueeze(0)


    def __len__(self) -> int:
        return len(self.fovs)

    def collate(self, samples: List[dict]) -> dict:
        return collate_fn(samples)


