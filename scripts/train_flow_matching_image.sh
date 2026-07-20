#!/usr/bin/env zsh
# Usage: bash scripts/train_flow_matching_image.sh abs=<true|false> cuda=<id|cuda:id>
#   abs=true  -> use absolute-action tasks (<task>_image_abs); abs=false -> <task>_image
#   cuda=0    -> cuda:0   (or pass a full device string, e.g. cuda=cuda:1)
set -e

export HYDRA_FULL_ERROR=1
export MUJOCO_GL=osmesa

# ---- arguments (key=value style): abs=true cuda=cuda:0 ----
abs_flag=false
cuda=1
if [ "$abs_flag" = true ]; then abs="_abs"; else abs=""; fi
if [[ "$cuda" == cuda:* ]]; then device="$cuda"; else device="cuda:${cuda}"; fi

# ---- fixed settings ----
task_config_name=can
debug=false
gen=flow_matching
config_name=train_flow_matching_unet_hybrid_workspace.yaml
num_inference_steps=16
n_envs=28

if [[ "$task_config_name" =~ ^(lift|can)$ ]]; then
    horizon=32
else
    horizon=48
fi

# tool_hang images are larger -> bigger crop
if [ "$task_config_name" = "tool_hang" ]; then
    crop_shape='[216,216]'
else
    crop_shape='[76,76]'
fi

if [ "$debug" = true ]; then
    subdir="debug/"
    seed_list="0"
    wandb offline
else
    subdir=""
    seed_list=(0)
    wandb online
fi

run_dir='outputs_HDD4/${task.name}_${task.dataset_type}${abs_tag}_reproduction/train_by_seed_'$gen'/'$subdir'seed${training.seed}_${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_cnn_${horizon}'
echo $run_dir
echo ${task_config_name}_image${abs}

for seed in $seed_list; do
    echo -e "\033[32m[Training ${task_config_name}_image${abs} | seed=${seed} | ${device}]\033[0m"
    python train.py --config-name=${config_name} \
        task=${task_config_name}_image${abs} \
        hydra.run.dir=$run_dir \
        logging.group='${task.name}_${task.dataset_type}_'$gen \
        logging.name='seed${training.seed}_${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_'$gen'_cnn_${horizon}' \
        training.debug=$debug \
        policy.num_inference_steps=$num_inference_steps \
        policy.crop_shape="$crop_shape" \
        training.seed=$seed training.device=$device \
        obs_as_global_cond=True \
        training.num_epochs=2000 \
        dataloader.num_workers=8 val_dataloader.num_workers=8 \
        task.env_runner.n_envs=$n_envs \
        dataloader.batch_size=64 \
        horizon=${horizon} task.dataset.horizon=${horizon} task.dataset.pad_after=$((horizon-1))
done
