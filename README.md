# CELL-Diff-VS
Diffusion based virtual staining with CELL-Diff

## Installation

To set up CELL-Diff-VS, begin by creating a conda environment:
```shell
conda create --name cell_diff_vs python=3.10
```

Activate the environment and run the installation script:
```shell
conda activate cell_diff_vs
bash install.sh
```

## Training

**Note: Edit the `run_*.sh` scripts to set the appropriate model and data paths before running.**

```shell
bash run_train.sh
```

## Validation

**Note: Edit the `run_*.sh` scripts to set the appropriate model and data paths before running.**

```shell
bash run_eval.sh
```

## Testing

**Note: Edit the `run_*.sh` scripts to set the appropriate model and data paths before running.**

For HEK293T Cell Dataset:
```shell
bash run_eval_hek.sh
```

For A549 Cell Dataset:
```shell
bash run_eval_a549.sh
```