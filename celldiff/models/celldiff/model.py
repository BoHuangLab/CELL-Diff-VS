import torch
import torch.nn as nn
import math

from celldiff.logging import logger
from .modules.patch_embed_3d import PatchEmbed3D

from .modules.diffusion import TimeStepEncoder, create_diffusion
from .modules.transformer import TransformerBlock, unpatchify, FinalLayer
from .modules.positional_embedding import get_3d_sincos_pos_embed
from .modules.simple_diffusion import ResnetBlock, Upsample, Downsample

from transformers import PreTrainedModel
from .config import CELLDiffLD3DVSConfig
from celldiff.pipeline.utils import CELLDiff3DVSOutput

from copy import deepcopy
from celldiff.models.vae_3d.vae_3d_model import VAE3DModel
from .modules import CondConvNet


class CELLDiffLD3DVSModel(PreTrainedModel):
    config_class = CELLDiffLD3DVSConfig

    def __init__(self, config: CELLDiffLD3DVSConfig):
        super().__init__(config)
        self.config = config

        self.net = UNetViT3D(config)

        if self.config.diffusion_pred_type == 'xstart':
            predict_xstart = True
        elif self.config.diffusion_pred_type == 'noise':
            predict_xstart = False
        self.diffusion = create_diffusion(
            timestep_respacing=config.timestep_respacing, 
            noise_schedule=config.ddpm_schedule, 
            learn_sigma=False, 
            image_d=config.model_input_size[-1], 
            predict_xstart=predict_xstart, 
        )

        self.vae_nucleus = self.initialize_vae()
        self.vae_membrane = self.initialize_vae()

        self.load_pretrained_weights(config, checkpoint_path=config.loadcheck_path)

        self.vae_nucleus = self.prepare_vae(self.vae_nucleus, self.config.vae_nucleus_loadcheck_path)
        self.vae_membrane = self.prepare_vae(self.vae_membrane, self.config.vae_membrane_loadcheck_path)

    def initialize_vae(self):
        vae_config = deepcopy(self.config)
        vae_config.ft = False
        vae_config.infer = False
        vae_config.vae_loadcheck_path = None
        vae = VAE3DModel(vae_config)

        return vae

    def prepare_vae(self, vae: VAE3DModel, checkpoint_path):
        if checkpoint_path is not None:
            vae_config = deepcopy(self.config)
            vae_config.infer = True
            vae.load_pretrained_weights(vae_config, checkpoint_path = checkpoint_path)

        for param in vae.parameters():
            param.requires_grad = False
        vae.eval()

        return vae

    def load_pretrained_weights(self, args, checkpoint_path):
        """
        Load pretrained weights from a given state_dict.
        """
        if args.ft or args.infer:
            if args.ft:
                logger.info(f"Finetune from checkpoint: {checkpoint_path}")
            else:
                logger.info(f"Infer from checkpoint: {checkpoint_path}")
                
            checkpoints_state = torch.load(checkpoint_path, map_location="cpu")
            if "model" in checkpoints_state:
                checkpoints_state = checkpoints_state["model"]
            elif "module" in checkpoints_state:
                checkpoints_state = checkpoints_state["module"]

            model_state_dict = self.state_dict()
            filtered_state_dict = {k: v for k, v in checkpoints_state.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}
            
            IncompatibleKeys = self.load_state_dict(filtered_state_dict, strict=False)
            IncompatibleKeys = IncompatibleKeys._asdict()

            missing_keys = []
            for keys in IncompatibleKeys["missing_keys"]:
                if keys.find("dummy") == -1:
                    missing_keys.append(keys)

            unexpected_keys = []
            for keys in IncompatibleKeys["unexpected_keys"]:
                if keys.find("dummy") == -1:
                    unexpected_keys.append(keys)

            if len(missing_keys) > 0:
                logger.info(
                    "Missing keys in {}: {}".format(
                        checkpoint_path,
                        missing_keys,
                    )
                )

            if len(unexpected_keys) > 0:
                logger.info(
                    "Unexpected keys {}: {}".format(
                        checkpoint_path,
                        unexpected_keys,
                    )
                )

    def config_optimizer(self):
        """
        Return the optimizer and learning rate scheduler for this model.

        Returns:
            tuple[Optimizer, LRScheduler]:
        """
        pass

    @torch.no_grad()
    def prepare_inputs(self, batched_data):
        phase = batched_data['phase']
        nucleus = batched_data['nucleus']
        membrane = batched_data['membrane']

        nucleus_latent = self.vae_nucleus.encode(nucleus).sample()
        membrane_latent = self.vae_membrane.encode(membrane).sample()

        return phase, nucleus_latent, membrane_latent

    def forward(self, batched_data, **kwargs):
        phase, nucleus_latent, membrane_latent = self.prepare_inputs(batched_data)

        source = phase
        target = torch.cat([nucleus_latent, membrane_latent], dim=1)

        # add noise to target
        time = torch.randint(
            0, 
            self.diffusion.num_timesteps, 
            (target.shape[0],), 
            device=target.device, 
        )

        noise = torch.randn_like(target)
        target_noisy = self.diffusion.q_sample(target, time, noise)

        model_time = self.diffusion._scale_timesteps(time)

        target_output = self.net(
            target_noisy, 
            source, 
            model_time, 
        )

        volume_mask = torch.ones_like(target)
        diff_loss_dict = self.diffusion.training_losses(
            target_output.to(torch.float32), 
            target_noisy.to(torch.float32), 
            target.to(torch.float32), 
            time, 
            noise, 
            volume_mask, 
        )

        loss = diff_loss_dict['loss'].mean()
        return CELLDiff3DVSOutput(loss=loss)

    def generate(self, phase, progress=True, sampling_strategy="ddim"):

        target_latent = torch.randn(
            phase.shape[0], 
            self.config.latent_channels * 2, 
            self.config.model_input_size[-3], 
            self.config.model_input_size[-2], 
            self.config.model_input_size[-1], 
        ).to(phase.device)
        indices = list(range(self.diffusion.num_timesteps))[::-1]

        volume_mask = torch.ones_like(target_latent).bool()

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            time = torch.tensor([i] * phase.shape[0], device=phase.device)
            with torch.no_grad():
                model_time = self.diffusion._scale_timesteps(time)
                target_latent_output = self.net(
                    target_latent, 
                    phase, 
                    model_time, 
                )
                if sampling_strategy == "ddpm":
                    out = self.diffusion.p_sample(
                        target_latent_output, 
                        target_latent, 
                        volume_mask, 
                        time, 
                        clip_denoised=False, 
                    )
                elif sampling_strategy == "ddim":
                    out = self.diffusion.ddim_sample(
                        target_latent_output,
                        target_latent,
                        volume_mask, 
                        time,
                        clip_denoised=False, 
                    )
                target_latent = out["sample"]

        nucleus_latent = target_latent[:, :self.config.latent_channels]
        membrane_latent = target_latent[:, self.config.latent_channels:]

        nucleus = self.vae_nucleus.decode(nucleus_latent).sample
        membrane = self.vae_membrane.decode(membrane_latent).sample

        return nucleus, membrane

    def embed(self, phase, nucleus=None, membrane=None):
        if nucleus is None:
            nucleus_latent = torch.randn(phase.shape[0], self.config.latent_channels, *self.config.model_input_size, device=phase.device)
        else:
            nucleus_latent = self.vae_nucleus.encode(nucleus).sample()

        if membrane is None:
            membrane_latent = torch.randn(phase.shape[0], self.config.latent_channels, *self.config.model_input_size, device=phase.device)
        else:
            membrane_latent = self.vae_membrane.encode(membrane).sample()

        if (nucleus is None) or (membrane is None):
            time = torch.tensor([self.diffusion.num_timesteps - 1] * phase.shape[0], device=phase.device)
        else:
            time = torch.tensor([0] * phase.shape[0], device=phase.device)

        model_time = self.diffusion._scale_timesteps(time)        

        source = phase
        target = torch.cat([nucleus_latent, membrane_latent], dim=1)

        self.net(target, source, model_time)

        return self.net.embed

class UNetViT3D(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args
        model_input_size = args.model_input_size
        input_spatial_size = args.input_spatial_size
        in_channels = args.latent_channels * 2
        cond_in_channels = 1
        cond_out_channels = args.cond_out_channels
        dims = args.dims
        num_res_block = args.num_res_block
        embed_dim = args.embed_dim
        num_heads = args.num_heads
        attn_drop = args.attn_drop
        depth = args.depth
        mlp_ratio = args.mlp_ratio
        patch_size = args.patch_size

        # Use ConvNet to process image
        self.inconv = nn.Conv3d(in_channels, dims[0], 3, 1, 1)
        self.cond_inconv = CondConvNet(cond_in_channels, cond_out_channels)

        self.downs = {}
        for i_level in range(len(num_res_block)): 
            for i_block in range(num_res_block[i_level]): 
                self.downs[str(i_level)+str(i_block)] = ResnetBlock(
                    dims[i_level], 
                    dims[i_level], 
                    embed_dim, 
                )

            self.downs['down'+str(i_level)] = Downsample(dims[i_level], dims[i_level+1])
        self.downs = nn.ModuleDict(self.downs)

        self.img_embedding = PatchEmbed3D(
            patch_size=patch_size, 
            in_chans=dims[-1], 
            embed_dim=embed_dim, 
            bias=True
        )

        # Use fixed sin-cos embedding:
        self.latent_size = [s // (2 ** len(num_res_block)) for s in model_input_size]
        self.latent_grid_size = [s // patch_size for s in self.latent_size]
        self.num_patches = math.prod(self.latent_grid_size)

        img_pos_embed = torch.from_numpy(get_3d_sincos_pos_embed(embed_dim, self.latent_grid_size)).float().unsqueeze(0)
        self.img_pos_embed = nn.Parameter(img_pos_embed, requires_grad=False)

        self.mids = nn.ModuleList([
            TransformerBlock(
                embed_dim, 
                num_heads, 
                mlp_ratio=mlp_ratio, 
                attn_drop=attn_drop, 
            ) for _ in range(depth)
        ])

        self.img_proj_out = FinalLayer(
            hidden_size=embed_dim, 
            patch_size=patch_size, 
            out_channels=dims[-1], 
        )

        self.ups = {}
        for i_level in reversed(range(len(num_res_block))):
            self.ups['up'+str(i_level)] = Upsample(dims[i_level+1], dims[i_level])
            for i_block in range(num_res_block[i_level]):
                self.ups[str(i_level)+str(i_block)] = ResnetBlock(
                    dims[i_level]*2, 
                    dims[i_level], 
                    embed_dim, 
                )
        self.ups = nn.ModuleDict(self.ups)
        
        self.outconv = nn.Conv3d(dims[0], in_channels, 3, 1, 1)
        self.t_embedding = TimeStepEncoder(embedding_dim=embed_dim)

        self.num_res_block = num_res_block
        self.in_channels = in_channels
        self.cond_in_channels = cond_in_channels
        self.dims = dims
        self.patch_size = patch_size
        self.model_input_size = model_input_size
        self.cond_input_size = input_spatial_size

    def forward(self, x, cond, t):

        assert list(x.shape[2:]) == self.model_input_size, \
            f"x spatial size {list(x.shape[2:])} does not match expected {self.model_input_size}"
        assert list(cond.shape[2:]) == self.cond_input_size, \
            f"cond spatial size {list(cond.shape[2:])} does not match expected {self.cond_input_size}"

        time_embeds = self.t_embedding(t)

        x = self.inconv(x)
        cond = self.cond_inconv(cond)
        concat_img = x + cond

        # downsample
        img_skips = []

        for i_level in range(len(self.num_res_block)):
            for i_block in range(self.num_res_block[i_level]):
                concat_img = self.downs[f"{i_level}{i_block}"](concat_img, time_embeds)
                img_skips.append(concat_img)

            concat_img = self.downs[f"down{i_level}"](concat_img)

        concat_img_embeds = self.img_embedding(concat_img) + self.img_pos_embed

        # mid
        for block in self.mids:
            concat_img_embeds = block(concat_img_embeds, time_embeds)

        concat_img = self.img_proj_out(concat_img_embeds, time_embeds)
        concat_img = unpatchify(concat_img, self.dims[-1], self.latent_grid_size, self.patch_size)

        self.embed = concat_img

        # upsample
        for i_level in reversed(range(len(self.num_res_block))):
            concat_img = self.ups['up' + str(i_level)](concat_img)

            for i_block in range(self.num_res_block[i_level]):
                skip_img = img_skips.pop()

                concat_img = torch.cat((concat_img, skip_img), dim=1)
                concat_img = self.ups[str(i_level) + str(i_block)](concat_img, time_embeds)

        output = self.outconv(concat_img)
        return output
