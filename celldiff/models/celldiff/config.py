# -*- coding: utf-8 -*-
from dataclasses import dataclass
from transformers import PretrainedConfig

@dataclass
class CELLDiffLD3DVSConfig(PretrainedConfig):
    model_type: str = 'diffusion'
    transformer_type: str = 'vit'

    # Dataset parameters
    data_path: str = ""
    split_key: str = 'train'
    data_aug: bool = False
    input_spatial_size: str = '32,256,256'

    phase_channel_name: str = 'phase'
    nucleus_channel_name: str = 'nuclues'
    membrane_channel_name: str = 'membrane'

    # Diffusion parameters
    num_timesteps: int = 1000
    ddpm_schedule: str = 'shifted_cos'
    diffusion_pred_type: str = 'noise'
    timestep_respacing: str = ""

    # Model parameters
    ## VAE
    in_channels: int = 1
    out_channels: int = 1
    num_down_blocks: int = 2
    latent_channels: int = 2
    vae_block_out_channels: str = '32,64'
    vae_nucleus_loadcheck_path: str = '.'
    vae_membrane_loadcheck_path: str = '.'

    ## CELL-Diff
    cond_out_channels: str = '32'
    model_input_size: str = '16,256,256'
    dims: str = '32,64,128'
    num_res_block: str = '2,2'
    embed_dim: int = 1280
    num_heads: int = 8
    attn_drop: float = 0.0
    depth: int = 8
    mlp_ratio: float = 4.0
    patch_size: int = 4

    # Training parameters
    loadcheck_path: str = '.'
    ft: bool = False
    infer: bool = False
    ifresume: bool = False

    output_dir: str = ""
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = 1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2

    num_train_epochs: int = 10
    fp16: bool = False
    bf16: bool = False
    logging_dir: str = ""
    logging_steps: int = 10
    max_steps: int = -1
    warmup_steps: int = 1000
    save_steps: int = 1000

    dataloader_num_workers: int = 16
    seed: int = 6

    # Evaluation
    num_samples: int = 1
    
    def __init__(self, **kwargs):
        # Use `super().__init__` to handle arguments from PretrainedConfig
        super().__init__(**kwargs)

        self.model_type = kwargs.get("model_type", self.model_type)
        self.transformer_type = kwargs.get("transformer_type", self.transformer_type)

        # Initialize all custom attributes from `kwargs` or defaults
        self.data_path = kwargs.get("data_path", self.data_path)
        self.split_key = kwargs.get("split_key", self.split_key)
        self.data_aug = kwargs.get("data_aug", self.data_aug)
        self.input_spatial_size = kwargs.get("input_spatial_size", self.input_spatial_size)
        if not isinstance(self.input_spatial_size, list):
            self.input_spatial_size = [int(a) for a in self.input_spatial_size.split(',')]

        self.phase_channel_name = kwargs.get("phase_channel_name", self.phase_channel_name)
        self.nucleus_channel_name = kwargs.get("nucleus_channel_name", self.nucleus_channel_name)
        self.membrane_channel_name = kwargs.get("membrane_channel_name", self.membrane_channel_name)

        self.num_timesteps = kwargs.get("num_timesteps", self.num_timesteps)
        self.ddpm_schedule = kwargs.get("ddpm_schedule", self.ddpm_schedule)
        self.diffusion_pred_type = kwargs.get("diffusion_pred_type", self.diffusion_pred_type)
        self.timestep_respacing = kwargs.get("timestep_respacing", self.timestep_respacing)

        self.in_channels = kwargs.get("in_channels", self.in_channels)
        self.out_channels = kwargs.get("out_channels", self.out_channels)
        self.num_down_blocks = kwargs.get("num_down_blocks", self.num_down_blocks)
        self.latent_channels = kwargs.get("latent_channels", self.latent_channels)
        self.vae_block_out_channels = kwargs.get("vae_block_out_channels", self.vae_block_out_channels)
        if not isinstance(self.vae_block_out_channels, list):
            self.vae_block_out_channels = [int(a) for a in self.vae_block_out_channels.split(',')]
        
        self.vae_nucleus_loadcheck_path = kwargs.get("vae_nucleus_loadcheck_path", self.vae_nucleus_loadcheck_path)
        self.vae_membrane_loadcheck_path = kwargs.get("vae_membrane_loadcheck_path", self.vae_membrane_loadcheck_path)

        self.cond_out_channels = kwargs.get("cond_out_channels", self.cond_out_channels)
        if not isinstance(self.cond_out_channels, list):
            self.cond_out_channels = [int(a) for a in self.cond_out_channels.split(',')]

        self.model_input_size = kwargs.get("model_input_size", self.model_input_size)
        if not isinstance(self.model_input_size, list):
            self.model_input_size = [int(a) for a in self.model_input_size.split(',')]
        self.dims = kwargs.get("dims", self.dims)
        if not isinstance(self.dims, list):
            self.dims = [int(a) for a in self.dims.split(',')]
        self.num_res_block = kwargs.get("num_res_block", self.num_res_block)
        if not isinstance(self.num_res_block, list):
            self.num_res_block = [int(a) for a in self.num_res_block.split(',')]
        self.embed_dim = kwargs.get("embed_dim", self.embed_dim)
        self.num_heads = kwargs.get("num_heads", self.num_heads)
        self.attn_drop = kwargs.get("attn_drop", self.attn_drop)
        self.depth = kwargs.get("depth", self.depth)
        self.mlp_ratio = kwargs.get("mlp_ratio", self.mlp_ratio)
        self.patch_size = kwargs.get("patch_size", self.patch_size)

        self.loadcheck_path = kwargs.get("loadcheck_path", self.loadcheck_path)
        self.ft = kwargs.get("ft", self.ft)
        self.infer = kwargs.get("infer", self.infer)
        self.ifresume = kwargs.get("ifresume", self.ifresume)

        self.output_dir = kwargs.get("output_dir", self.output_dir)
        self.learning_rate = kwargs.get("learning_rate", self.learning_rate)
        self.weight_decay = kwargs.get("weight_decay", self.weight_decay)
        self.gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps", self.gradient_accumulation_steps)
        self.per_device_train_batch_size = kwargs.get("per_device_train_batch_size", self.per_device_train_batch_size)
        self.per_device_eval_batch_size = kwargs.get("per_device_eval_batch_size", self.per_device_eval_batch_size)

        self.num_train_epochs = kwargs.get("num_train_epochs", self.num_train_epochs)
        self.fp16 = kwargs.get("fp16", self.fp16)
        self.bf16 = kwargs.get("bf16", self.bf16)
        self.logging_dir = kwargs.get("logging_dir", self.logging_dir)
        self.logging_steps = kwargs.get("logging_steps", self.logging_steps)
        self.max_steps = kwargs.get("max_steps", self.max_steps)
        self.warmup_steps = kwargs.get("warmup_steps", self.warmup_steps)
        self.save_steps = kwargs.get("save_steps", self.save_steps)

        self.dataloader_num_workers = kwargs.get("dataloader_num_workers", self.dataloader_num_workers)
        self.seed = kwargs.get("seed", self.seed)

        self.num_samples = kwargs.get("num_samples", self.num_samples)