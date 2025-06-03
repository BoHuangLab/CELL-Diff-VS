# -*- coding: utf-8 -*-
from typing import List
import torch

def collate_fn(samples: List[dict]):
    batch = dict()
        
    # Get all keys from the first sample
    keys = samples[0].keys()

    for key in keys:
        # Stack each tensor along a new batch dimension
        batch[key] = torch.cat([s[key].unsqueeze(0) for s in samples])

    return {'batched_data': batch}