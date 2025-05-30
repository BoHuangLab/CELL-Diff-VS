# Using Weights & Biases (WandB) for experiment tracking (optional)
# export WANDB_RUN_NAME=       # (Optional) Name of the current run (visible in the WandB dashboard)
# export WANDB_API_KEY=        # (Optional) Your WandB API key to authenticate logging
# export WANDB_PROJECT=        # (Optional) WandB project name to group related experiments

# Set the output directory for training results
export output_dir=pretrain/pt_celldiff_vscyto3d

# Set the path to the training dataset
export data_path=/hpc/websites/public.czbiohub.org/comp.micro/viscy/VS_datasets/VSCyto3D/train/raw-and-reconstructed.zarr

# Set the path to the pretrained VAE checkpoint
export vae_nucleus_loadcheck_path=pretrained_models/vaes/nucleus_vae.bin
export vae_membrane_loadcheck_path=pretrained_models/vaes/membrane_vae.bin

bash scripts/celldiff/pretrain.sh