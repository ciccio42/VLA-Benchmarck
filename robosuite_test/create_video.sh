#!/bin/bash
#SBATCH -A hpc_default
#SBATCH --exclude=tnode[01-17]
#SBATCH --exclude=gnode14
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL

export MUJOCO_PY_MUJOCO_PATH="/home/rsofnc000/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
PKL_PATH=/home/rsofnc000/checkpoint_save_folder/tiny_vla/post_processed_tiny_vla_llava_pythia_lora_ur5e_pick_place_delta_removed_0_5_10_15_lora_r_128_processed/checkpoint-40000/rollout_pick_place_0_False_obj_set_-1_change_command_False

srun python create_video.py \
    --path_to_pkl $PKL_PATH \
    --output_dir $PKL_PATH/video
