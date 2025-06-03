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
import umap
import matplotlib.pyplot as plt
from tqdm import tqdm


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

    all_embeds = []

    for i, data in enumerate(tqdm(valset, desc="Processing samples")):
        name = data['name']
        phase = data['phase'].unsqueeze(0).to(device)
        nucleus = data['nucleus'].unsqueeze(0).to(device)
        membrane = data['membrane'].unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model.embed(
                phase, 
                nucleus, 
                membrane
            )
            all_embeds.append(embedding.detach().cpu().view(1, -1))

    all_embeds = torch.cat(all_embeds, dim=0)

    umap_model = umap.UMAP(n_components=2, random_state=42, low_memory=False, n_neighbors=15, min_dist=0.1,verbose=True)
    umap_all_embeds = umap_model.fit_transform(all_embeds)

    plt.figure(figsize=(8, 8))
    plt.scatter(umap_all_embeds[:, 0], umap_all_embeds[:, 1], s=5, alpha=0.7)
    plt.title("UMAP Projection of Embeddings")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "umap_projection.png"
    plt.savefig(fig_path, dpi=300)
    print(f"UMAP projection saved to {fig_path}")

if __name__ == "__main__":
    main()