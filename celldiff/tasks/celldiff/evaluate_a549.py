# -*- coding: utf-8 -*-
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from celldiff.data.a549_data.dataset import ZarrTestDataset
from celldiff.models.celldiff.config import CELLDiffLD3DVSConfig
from celldiff.models.celldiff.model import CELLDiffLD3DVSModel

from celldiff.utils.cli_utils import cli

import numpy as np
from iohub.ngff import open_ome_zarr
from tqdm import tqdm

def _crop_fov(phase: torch.Tensor, fov_key: str, size: tuple[int, int, int]) -> torch.Tensor:
    """
    Apply the hard coded crop windows.
    """
    zs, ys, xs = size
    if fov_key == "B/1/000001":
        z0, y0, x0 = 20, 154, 476
    elif fov_key == "B/2/001000":
        z0, y0, x0 = 20, 740, 212
    else:
        z0 = y0 = x0 = 0
    return phase[:, z0:z0 + zs, y0:y0 + ys, x0:x0 + xs]

@cli(CELLDiffLD3DVSConfig)
def main(args) -> None:
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"

    config = CELLDiffLD3DVSConfig(**vars(args))

    valset = ZarrTestDataset(config)
    model = CELLDiffLD3DVSModel(config=config)

    input_spatial_size = config.input_spatial_size

    model.to(device)
    model.eval()

    output_dir = config.output_dir
    channel_names = ['Phase', 'Nuclei-prediction', 'Membrane-prediction']
    output_dataset = open_ome_zarr(output_dir, mode="w", layout='hcs', channel_names=channel_names)

    for i, data in enumerate(valset):
        phase = data['phase']
        fov_path = data['fov_path']

        phase = _crop_fov(phase, fov_path, input_spatial_size)
        
        output_data = torch.zeros(
            phase.shape[0], 
            3, 
            phase.shape[1], 
            phase.shape[2], 
            phase.shape[3], 
        )

        for t in tqdm(range(phase.shape[0])):
            phase_t = phase[t].unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                nucleus_gen, membrane_gen = model.generate(
                    phase_t, 
                    sampling_strategy="ddim", 
                    progress=False, 
                )
            
            sample = torch.cat([nucleus_gen, membrane_gen], dim=1)
            sample = sample.clamp(-1, 1)

            phase_t = (phase_t + 1) / 2
            phase_t = phase_t.detach().cpu()
            sample = (sample + 1) / 2
            sample = sample.detach().cpu()
            output_data[t] = torch.cat([phase_t, sample], dim=1).squeeze(0)

        output_data = output_data.numpy().astype(np.float32)
        row, col, fov = fov_path.split("/")
        out_pos = output_dataset.create_position(row, col, fov)
        out_pos.create_image("0", output_data)

if __name__ == "__main__":
    main()
