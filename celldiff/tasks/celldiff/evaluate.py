# -*- coding: utf-8 -*-
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from pathlib import Path

from celldiff.data.vscyto3d_npy_data.dataset import VSCyto3DTestNPYDataset
from celldiff.models.celldiff.config import CELLDiffLD3DVSConfig
from celldiff.models.celldiff.model import CELLDiffLD3DVSModel
from celldiff.utils.cli_utils import cli

import numpy as np
import tifffile as tiff

def save_tif(image, output_path):
    tensor_np = image.squeeze(0).cpu().numpy()
    tensor_np = np.transpose(tensor_np, (1, 0, 2, 3))

    tensor_np = (tensor_np + 1) / 2.0
    tensor_np = tensor_np * 65535

    tensor_np = tensor_np.astype(np.uint16)    
    tiff.imwrite(output_path, tensor_np, imagej=True)

@cli(CELLDiffLD3DVSConfig)
def main(args) -> None:
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"

    config = CELLDiffLD3DVSConfig(**vars(args))

    valset = VSCyto3DTestNPYDataset(config)
    model = CELLDiffLD3DVSModel(config=config)

    model.to(device)
    model.eval()

    output_dir = Path(config.output_dir)
    output_dir = output_dir

    num_samples = config.num_samples

    for i, data in enumerate(valset):

        name = data['name']
        phase = data['phase'].unsqueeze(0).to(device)
        nucleus = data['nucleus'].unsqueeze(0).to(device)
        membrane = data['membrane'].unsqueeze(0).to(device)

        target = torch.cat([nucleus, membrane], dim=1)

        save_dir = output_dir / f'{name}'
        save_dir.mkdir(parents=True, exist_ok=True)

        for j in range(num_samples):
            with torch.no_grad():
                nucleus, membrane = model.generate(
                    phase, 
                    sampling_strategy="ddim", 
                    progress=True, 
                )
            
            sample = torch.cat([nucleus, membrane], dim=1)
            sample = sample.clamp(-1, 1)
            save_tif(sample, save_dir/'prediction_{:02d}.tif'.format(j+1))

        save_tif(phase, save_dir/'phase.tif')
        save_tif(target, save_dir/'target.tif')

if __name__ == "__main__":
    main()
