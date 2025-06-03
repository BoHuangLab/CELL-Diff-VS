# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import torch

from typing import List
from torch.utils.data import Dataset

from .collater import collate_fn

from torchvision import transforms
from .img_utils import RandomRotation, crop_or_pad, normalize
from iohub.ngff import open_ome_zarr
import zarr


class VSCyto3DNPYDataset(Dataset):
    def __init__(self, args, split_key) -> None:
        super().__init__()
        self.args = args

        self.split_key = split_key
        self.data_path = self.args.data_path

        with open(os.path.join(self.data_path, f'keys/{self.split_key}_keys.json')) as f:
            self.data_files = json.load(f)

        self.spatial_size = self.args.input_spatial_size

        self.phase_global_min = -0.06567
        self.phase_global_max = 0.07901

        self.nucleus_global_min = 0
        self.nucleus_global_max = 11520.41875
        
        self.membrane_global_min = 0
        self.membrane_global_max = 528521.60937

    def __getitem__(self, index: int) -> dict:        
        data_file = self.data_files[index]
        data = np.load(os.path.join(self.data_path, f'data/{data_file}.npy'))
        
        item = {}

        item['phase'], item['nucleus'], item['membrane'], item['volume_mask'] = self.get_img(data)
        
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

        img = torch.stack([phase, nucleus, membrane], dim=0)
        img, volume_mask = crop_or_pad(img, self.spatial_size[2], self.spatial_size[0])

        t_forms = []

        if self.args.data_aug:
            img = torch.cat([img, volume_mask], dim=0)
            t_forms.append(transforms.RandomHorizontalFlip(p=0.5))
            t_forms.append(RandomRotation([0, 90, 180, 270]))
            t_forms = transforms.Compose(t_forms)
            phase, nucleus, membrane, volume_mask = t_forms(img)
        else:
            phase, nucleus, membrane = img
            volume_mask = volume_mask.squeeze(0)

        phase = phase * 2 - 1
        nucleus = nucleus * 2 - 1
        membrane = membrane * 2 - 1
        
        return phase.unsqueeze(0), nucleus.unsqueeze(0), membrane.unsqueeze(0), volume_mask.unsqueeze(0).round().bool()

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

        self.spatial_size = args.input_spatial_size

    def __getitem__(self, index: int) -> dict:
        fov = self.fovs[index]
        data = self.data[0][fov][0][0][0, 3:]
        
        item = {}
        item['phase'], item['nucleus'], item['membrane'], item['volume_mask'] = self.get_img(data)
        
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

        img = torch.stack([phase, nucleus, membrane], dim=0)
        img, volume_mask = crop_or_pad(img, self.spatial_size[2], self.spatial_size[0])

        t_forms = []

        if self.args.data_aug:
            img = torch.cat([img, volume_mask], dim=0)
            t_forms.append(transforms.RandomHorizontalFlip(p=0.5))
            t_forms.append(RandomRotation([0, 90, 180, 270]))
            t_forms = transforms.Compose(t_forms)
            phase, nucleus, membrane, volume_mask = t_forms(img)
        else:
            phase, nucleus, membrane = img
            volume_mask = volume_mask.squeeze(0)

        phase = phase * 2 - 1
        nucleus = nucleus * 2 - 1
        membrane = membrane * 2 - 1
        
        return phase.unsqueeze(0), nucleus.unsqueeze(0), membrane.unsqueeze(0), volume_mask.unsqueeze(0).round().bool()

    def __len__(self) -> int:
        return len(self.fovs)

    def collate(self, samples: List[dict]) -> dict:
        return collate_fn(samples)


class VSCyto3DZarrTestDataset(Dataset):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

        self.data_path = self.args.data_path

        plate = open_ome_zarr(self.data_path, mode="r")
        self.fovs = [pos for _, pos in plate.positions()]

        self.spatial_size = self.args.input_spatial_size

        self.phase_global_min = 588.39603
        self.phase_global_max = 37762.16797

        self.nucleus_global_min = 0
        self.nucleus_global_max = 11520.41875
        
        self.membrane_global_min = 0
        self.membrane_global_max = 528521.60937

        self.phase_channel_name = args.phase_channel_name
        self.nucleus_channel_name = args.nucleus_channel_name
        self.membrane_channel_name = args.membrane_channel_name

    def __getitem__(self, index: int) -> dict:
        pos = self.fovs[index]
        data = pos.data

        phase_channel_idx = pos.get_channel_index(self.phase_channel_name)
        nucleus_channel_idx = pos.get_channel_index(self.nucleus_channel_name)
        membrane_channel_idx = pos.get_channel_index(self.membrane_channel_name)

        phase = data[:, phase_channel_idx]
        nucleus = data[:, nucleus_channel_idx]
        membrane = data[:, membrane_channel_idx]

        item = {}

        item['phase'], item['nucleus'], item['membrane'], item['volume_mask'] = self.get_img(phase, nucleus, membrane)
        
        return item

    def get_img(self, phase, nucleus, membrane):        
        phase = normalize(phase, self.phase_global_min, self.phase_global_max)
        nucleus = normalize(nucleus, self.nucleus_global_min, self.nucleus_global_max)
        membrane = normalize(membrane, self.membrane_global_min, self.membrane_global_max)

        phase = - 1.6770367287483092 * phase + 0.6080998924065107
        phase = phase.clip(0, 1)

        phase = torch.from_numpy(phase)
        nucleus = torch.from_numpy(nucleus)
        membrane = torch.from_numpy(membrane)

        img = torch.cat([phase, nucleus, membrane], dim=0)
        img, volume_mask = crop_or_pad(img, self.spatial_size[2], self.spatial_size[0])

        phase = img[:phase.shape[0]]
        nucleus = img[phase.shape[0]:phase.shape[0]+nucleus.shape[0]]
        membrane = img[nucleus.shape[0]+phase.shape[0]:]

        volume_mask = volume_mask.squeeze(0)

        phase = phase * 2 - 1
        nucleus = nucleus * 2 - 1
        membrane = membrane * 2 - 1
        
        return phase, nucleus, membrane, volume_mask.unsqueeze(0).round().bool()

    def __len__(self) -> int:
        return len(self.fovs)

    def collate(self, samples: List[dict]) -> dict:
        return collate_fn(samples)
