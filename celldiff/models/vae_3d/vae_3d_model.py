import os
import torch
import torch.nn as nn
from .modules.autoencoders import Autoencoder3DKL
from .vae_3d_config import VAE3DConfig
from transformers import PreTrainedModel
from celldiff.pipeline.utils import VAEOutput
from celldiff.logging import logger


class VAE3DModel(PreTrainedModel):
    config_class = VAE3DConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        self.num_down_blocks = config.num_down_blocks 
        self.num_up_blocks = self.num_down_blocks

        # Initialize Autoencoder3DKL
        self.vae = Autoencoder3DKL(
            in_channels=config.in_channels, 
            out_channels=config.out_channels, 
            num_down_blocks=self.num_down_blocks, 
            num_up_blocks=self.num_up_blocks, 
            block_out_channels=config.vae_block_out_channels, 
            latent_channels=config.latent_channels, # Latent space dimensions
        )

        self.load_pretrained_weights(config, checkpoint_path=config.vae_loadcheck_path)

    def load_pretrained_weights(self, config, checkpoint_path):
        """
        Load pretrained weights from a given state_dict.
        """
        if config.ft or config.infer:
            if config.ft:
                logger.info(f"Finetune from checkpoint: {checkpoint_path}")
            else:
                logger.info(f"Infer from checkpoint: {checkpoint_path}")
                
            if os.path.splitext(checkpoint_path)[1] == '.safetensors':
                from safetensors.torch import load_file
                checkpoints_state = load_file(checkpoint_path)
            else:
                checkpoints_state = torch.load(checkpoint_path, map_location="cpu")

            if "model" in checkpoints_state:
                checkpoints_state = checkpoints_state["model"]
            elif "module" in checkpoints_state:
                checkpoints_state = checkpoints_state["module"]

            model_state_dict = self.state_dict()
            filtered_state_dict = {k: v for k, v in checkpoints_state.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}

            IncompatibleKeys = self.load_state_dict(filtered_state_dict, strict=False)
            # IncompatibleKeys = self.load_state_dict(checkpoints_state, strict=False)
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

    def encode(self, x):
        """Encodes input into latent space."""
        return self.vae.encode(x).latent_dist

    def decode(self, latents):
        """Decodes latent space into reconstructed input."""
        return self.vae.decode(latents)

    def forward(self, batched_data):
        x = batched_data['data']

        """Forward pass through the VAE."""
        latent_dist = self.encode(x)
        latents = latent_dist.sample()
        recon_x = self.decode(latents).sample

        total_loss, recon_loss, kl_loss = self.compute_loss(x, recon_x, latent_dist)

        return VAEOutput(total_loss, recon_loss, kl_loss)

    def compute_loss(self, x, recon_x, latent_dist):
        """Compute reconstruction and KL divergence loss."""
        if self.config.vae_recon_loss_type == 'mse':
            recon_loss = nn.MSELoss()(recon_x, x)
        elif self.config.vae_recon_loss_type == 'poisson':
            x = x.clip(-1, 1)
            recon_x = recon_x.clip(-1, 1)
            peak = self.config.poisson_peak if hasattr(self.config, 'poisson_peak') else 1.0
            target = (x + 1) / 2.0 * peak
            lam = (recon_x + 1) / 2.0 * peak
            recon_loss = torch.mean(lam - target * torch.log(lam + 1e-8))

        kl_loss = -0.5 * torch.mean(1 + latent_dist.logvar - latent_dist.mean.pow(2) - latent_dist.logvar.exp())
        total_loss = self.config.recon_loss_coeff * recon_loss + self.config.kl_loss_coeff * kl_loss
        return total_loss, recon_loss, kl_loss

    def sample(self, num_samples=1, latent_size=32, device="cpu"):
        """
        Generate samples from the latent space.

        Args:
            num_samples (int): Number of samples to generate.
            device (str): Device to perform sampling on.

        Returns:
            torch.Tensor: Generated images.
        """
        # Sample from a standard normal distribution in latent space
        latents = torch.randn((num_samples, self.config.latent_channels, latent_size, latent_size, latent_size), device=device)  # Shape matches latent dimensions

        # Decode latents to generate images
        with torch.no_grad():
            generated_images = self.decode(latents).sample

        return generated_images
    
    # def reconstruct(self, x):
    #     latent_dist = self.encode(x)
    #     latents = latent_dist.sample()  # Reparameterization trick
    #     recon_x = self.decode(latents).sample

    #     return recon_x

    def reconstruct(self, x):
        """
        Reconstruct input `x` using VAE with tiled decoding along the last two dims.
        Splits the latent tensor into 2x2 tiles (4 parts), decodes them separately,
        and stitches the reconstruction back together.
        """
        latent_dist = self.encode(x)
        latents = latent_dist.sample()  # shape: (B, C, D, H, W) or similar
        B, C, D, H, W = latents.shape

        # Ensure H and W are divisible by 2
        assert H % 2 == 0 and W % 2 == 0, "H and W must be divisible by 2 for 2x2 tiling."

        # Split latents into 2x2 grid
        h_half, w_half = H // 2, W // 2

        latents_00 = latents[..., :h_half, :w_half]   # top-left
        latents_01 = latents[..., :h_half, w_half:]   # top-right
        latents_10 = latents[..., h_half:, :w_half]   # bottom-left
        latents_11 = latents[..., h_half:, w_half:]   # bottom-right

        # Decode each tile
        recon_00 = self.decode(latents_00).sample
        recon_01 = self.decode(latents_01).sample
        recon_10 = self.decode(latents_10).sample
        recon_11 = self.decode(latents_11).sample

        # Stitch back together
        top = torch.cat([recon_00, recon_01], dim=-1)
        bottom = torch.cat([recon_10, recon_11], dim=-1)
        recon_x = torch.cat([top, bottom], dim=-2)

        return recon_x
