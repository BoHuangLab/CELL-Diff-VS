# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import torch

from typing import List
from torch.utils.data import Dataset

from .collater import collate_fn

from glob import glob

from torchvision import transforms
from .img_utils import (
    RandomRotation, 
    normalize, 
    DepthCrop, 
    RandomGaussianNoise, 
    RandomAdjustContrast, 
    RandomAdjustGamma, 
    RandomCropResize, 
)

class VSCyto3DNPYDataset(Dataset):
    def __init__(self, args, split_key) -> None:
        super().__init__()
        self.args = args

        self.split_key = split_key
        self.data_path = self.args.data_path

        with open(os.path.join(self.data_path, f'keys/{self.split_key}_keys.json')) as f:
            self.data_files = json.load(f)

        self.phase_global_min = -0.06567
        self.phase_global_max = 0.07901

        self.nucleus_global_min = 0
        self.nucleus_global_max = 11520.41875
        
        self.membrane_global_min = 0
        self.membrane_global_max = 528521.60937

        self.spatial_size = self.args.input_spatial_size
        self.data_aug = self.args.data_aug

    def __getitem__(self, index: int) -> dict:        
        data_file = self.data_files[index]
        data = np.load(os.path.join(self.data_path, f'data/{data_file}.npy'))
        
        item = {}

        item['name'] = data_file
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
            t_forms.append(RandomCropResize((self.spatial_size[-1], self.spatial_size[-2]), rescale_range=(0.5, 2)))
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
        return len(self.data_files)

    def collate(self, samples: List[dict]) -> dict:
        return collate_fn(samples)


class VSCyto3DSepNPYDataset(VSCyto3DNPYDataset):
    def __init__(self, args, split_key) -> None:
        super().__init__(args, split_key)
        self.data_type = args.data_type

    def __getitem__(self, index: int) -> dict:        
        data_file = self.data_files[index]
        data = np.load(os.path.join(self.data_path, f'data/{data_file}.npy'))
        
        item = {}

        phase, nucleus, membrane, volume_mask = self.get_img(data)

        if self.data_type == 'phase':
            item['data'] = phase
        elif self.data_type == 'nucleus':
            item['data'] = nucleus
        elif self.data_type == 'membrane':
            item['data'] = membrane

        item['volume_mask'] = volume_mask
        
        return item


class VSCyto3DSliceNPYDataset(VSCyto3DNPYDataset):
    def __init__(self, args, split_key) -> None:
        super().__init__(args, split_key)
        self.data_type = args.data_type

        self.fovs_slices = []
        for fov in self.fovs:
            data = self.data[0][fov][0][0]
            num_slices = data.shape[2]

            fov_slices = [f'{fov}_{num_slice}' for num_slice in range(num_slices)]
            self.fovs_slices = self.fovs_slices + fov_slices

    def __getitem__(self, index: int) -> dict:        
        data_file = self.data_files[index]
        data = np.load(os.path.join(self.data_path, f'data/{data_file}.npy'))
        
        item = {}

        phase, nucleus, membrane, volume_mask = self.get_img(data)

        if self.data_type == 'phase':
            item['data'] = phase
        elif self.data_type == 'nucleus':
            item['data'] = nucleus
        elif self.data_type == 'membrane':
            item['data'] = membrane

        item['volume_mask'] = volume_mask
        
        return item
    

class VSCyto3DTestNPYDataset(VSCyto3DNPYDataset):
    def __init__(self, args) -> None:
        self.args = args

        self.data_path = self.args.data_path
        self.data_files = sorted([
            os.path.splitext(f)[0] for f in os.listdir(self.data_path)
            if os.path.isfile(os.path.join(self.data_path, f)) and f.endswith('.npy')
        ])

        self.phase_global_min = -0.06567
        self.phase_global_max = 0.07901

        self.nucleus_global_min = 0
        self.nucleus_global_max = 11520.41875
        
        self.membrane_global_min = 0
        self.membrane_global_max = 528521.60937

        self.spatial_size = self.args.input_spatial_size
        self.data_aug = self.args.data_aug

    def __getitem__(self, index: int) -> dict:        
        data_file = self.data_files[index]
        data = np.load(os.path.join(self.data_path, f'{data_file}.npy'))
        
        item = {}

        item['name'] = data_file
        item['phase'], item['nucleus'], item['membrane'] = self.get_img(data)
        
        return item