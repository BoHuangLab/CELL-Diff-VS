ulimit -c unlimited
[ -z "${output_dir}" ] && output_dir=./output

# Dataset
[ -z "${data_path}" ] && data_path=data/val
[ -z "${input_spatial_size}" ] && input_spatial_size='32,512,512'

# DDPM
[ -z "${num_timesteps}" ] && num_timesteps=1000
[ -z "${ddpm_schedule}" ] && ddpm_schedule=shifted_cos
[ -z "${diffusion_pred_type}" ] && diffusion_pred_type=xstart

# Model
## VAE
[ -z "${in_channels}" ] && in_channels=1
[ -z "${out_channels}" ] && out_channels=1
[ -z "${num_down_blocks}" ] && num_down_blocks=2
[ -z "${latent_channels}" ] && latent_channels=2
[ -z "${vae_block_out_channels}" ] && vae_block_out_channels='32,64'
[ -z "${vae_nucleus_loadcheck_path}" ] && vae_nucleus_loadcheck_path=.
[ -z "${vae_membrane_loadcheck_path}" ] && vae_membrane_loadcheck_path=.

## CELL-Diff
[ -z "${cond_out_channels}" ] && cond_out_channels='64'
[ -z "${model_input_size}" ] && model_input_size='16,256,256'
[ -z "${dims}" ] && dims='64,128,256'
[ -z "${num_res_block}" ] && num_res_block='2,2'
[ -z "${embed_dim}" ] && embed_dim=1280
[ -z "${num_heads}" ] && num_heads=8
[ -z "${attn_drop}" ] && attn_drop=0.0
[ -z "${depth}" ] && depth=8
[ -z "${mlp_ratio}" ] && mlp_ratio=4.0
[ -z "${patch_size}" ] && patch_size=4

# Training
[ -z "${loadcheck_path}" ] && loadcheck_path=.
[ -z "${learning_rate}" ] && learning_rate=3e-4
[ -z "${weight_decay}" ] && weight_decay=0.0
[ -z "${gradient_accumulation_steps}" ] && gradient_accumulation_steps=4
[ -z "${per_device_train_batch_size}" ] && per_device_train_batch_size=8
[ -z "${per_device_eval_batch_size}" ] && per_device_eval_batch_size=128

[ -z "${num_train_epochs}" ] && num_train_epochs=5000
[ -z "${logging_dir}" ] && logging_dir=$output_dir
[ -z "${logging_steps}" ] && logging_steps=100
[ -z "${warmup_steps}" ] && warmup_steps=1000
[ -z "${max_steps}" ] && max_steps=50000
[ -z "${save_steps}" ] && save_steps=1000

# Evaluation
[ -z "${timestep_respacing}" ] && timestep_respacing="ddim100"


python celldiff/tasks/celldiff/evaluate.py \
            --output_dir $output_dir \
            --data_path $data_path \
            --input_spatial_size $input_spatial_size \
            --num_timesteps $num_timesteps \
            --ddpm_schedule $ddpm_schedule \
            --diffusion_pred_type $diffusion_pred_type \
            --in_channels $in_channels \
            --out_channels $out_channels \
            --num_down_blocks $num_down_blocks \
            --latent_channels $latent_channels \
            --vae_block_out_channels $vae_block_out_channels \
            --vae_nucleus_loadcheck_path $vae_nucleus_loadcheck_path \
            --vae_membrane_loadcheck_path $vae_membrane_loadcheck_path \
            --cond_out_channels $cond_out_channels \
            --model_input_size $model_input_size \
            --dims $dims \
            --num_res_block $num_res_block \
            --embed_dim $embed_dim \
            --num_heads $num_heads \
            --attn_drop $attn_drop \
            --depth $depth \
            --mlp_ratio $mlp_ratio \
            --patch_size $patch_size \
            --loadcheck_path $loadcheck_path \
            --learning_rate $learning_rate \
            --weight_decay $weight_decay \
            --gradient_accumulation_steps $gradient_accumulation_steps \
            --per_device_train_batch_size $per_device_train_batch_size \
            --per_device_eval_batch_size $per_device_eval_batch_size \
            --num_train_epochs $num_train_epochs \
            --logging_dir $logging_dir \
            --logging_steps $logging_steps \
            --warmup_steps $warmup_steps \
            --max_steps $max_steps \
            --save_steps $save_steps \
            --seed 6 \
            --timestep_respacing $timestep_respacing \
            --infer \