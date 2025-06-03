# -*- coding: utf-8 -*-
from typing import List
import torch

# def collate_fn(samples: List[dict]):
#     batch = dict()

#     batch["phase"] = torch.cat(
#         [s["phase"].unsqueeze(0) for s in samples]
#     )

#     batch["nucleus"] = torch.cat(
#         [s["nucleus"].unsqueeze(0) for s in samples]
#     )

#     batch["membrane"] = torch.cat(
#         [s["membrane"].unsqueeze(0) for s in samples]
#     )

#     batch["volume_mask"] = torch.cat(
#         [s["volume_mask"].unsqueeze(0) for s in samples]
#     )

#     return {'batched_data': batch}

def collate_fn(samples: List[dict]):
    batch = dict()
        
    # Get all keys from the first sample
    keys = samples[0].keys()

    for key in keys:
        # Stack each tensor along a new batch dimension
        batch[key] = torch.cat([s[key].unsqueeze(0) for s in samples])

    return {'batched_data': batch}