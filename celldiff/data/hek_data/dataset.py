# -*- coding: utf-8 -*-
import torch
from typing import List
from torch.utils.data import Dataset
from .collater import collate_fn
from .img_utils import normalize
from iohub.ngff import open_ome_zarr

class ZarrTestDataset(Dataset):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

        self.data_path = self.args.data_path
        self.fov_paths = ["0/HIST2H2BE/000001"]

        self.phase_global_min = -0.06567
        self.phase_global_max = 0.07901

        self.nucleus_global_min = 0
        self.nucleus_global_max = 11520.41875

        self.membrane_global_min = 0
        self.membrane_global_max = 528521.60937

        self.spatial_size = self.args.input_spatial_size

    def __getitem__(self, index: int) -> dict:

        with open_ome_zarr(self.data_path, mode="r") as ds:
            fov_path = self.fov_paths[index]
            data = ds[f"{fov_path}/0"]

        phase = data[:, 0]
        item = {}

        item['phase'] = self.get_img(phase)
        item['fov_path'] = fov_path

        return item

    def get_img(self, phase):
        phase = normalize(phase, self.phase_global_min, self.phase_global_max)

        phase = torch.from_numpy(phase)
        phase = phase * 2 - 1

        return phase

    def __len__(self) -> int:
        return len(self.fov_paths)

    def collate(self, samples: List[dict]) -> dict:
        return collate_fn(samples)
