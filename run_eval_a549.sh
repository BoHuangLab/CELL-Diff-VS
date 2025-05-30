# Set the output directory
export output_dir=./prediction/a549/output.zarr

# Set the path to the validation dataset
export data_path=/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2025_04_17_A549_H2B_CAAX_DENV/1-preprocess/label-free/2-stabilize/phase/2025_04_17_A549_H2B_CAAX_DENV.zarr

# Set the path to the pretrained CELL-Diff checkpoint
export loadcheck_path=pretrained_models/celldiff/vscyto3d_pretrained.bin

# Set the path to the pretrained VAE checkpoint
export vae_nucleus_loadcheck_path=pretrained_models/vaes/nucleus_vae.bin
export vae_membrane_loadcheck_path=pretrained_models/vaes/membrane_vae.bin

bash scripts/celldiff/evaluate_a549.sh