# Set the output directory
export output_dir=./prediction/hek/output.zarr

# Set the path to the validation dataset
export data_path=/hpc/projects/virtual_staining/datasets/mehta-lab/DynaCell/temp_figure_4.zarr

# Set the path to the pretrained CELL-Diff checkpoint
export loadcheck_path=pretrained_models/celldiff/vscyto3d_pretrained.bin

# Set the path to the pretrained VAE checkpoint
export vae_nucleus_loadcheck_path=pretrained_models/vaes/nucleus_vae.bin
export vae_membrane_loadcheck_path=pretrained_models/vaes/membrane_vae.bin

bash scripts/celldiff/evaluate_hek.sh