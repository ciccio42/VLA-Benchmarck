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

# DATASET_NAME=${1:-"ur5e_pick_place_rm_central_spawn"}
# CKPT_NUMBER=${2:-"45000"}
# CHANGE_SPAWN_REGIONS=${3:-"False"}
# RUN=${4:-"1"}
# CHUNK_SIZE=${5:-"1"}
# TASK_SUITE_NAME=${6:-"ur5e_pick_place_rm_central_spawn"}
# OBJECT_SET=${7:-"-1"}
# CHANGE_COMMAND=${8:-"False"}
# OUTPUT_FOLDER=${9:-"/home/rsofnc000/checkpoint_save_folder/open_vla"}

# ID_NOTE=${DATASET_NAME}_parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img-gripper_img-proprio
# MODEL_PATH=${OUTPUT_FOLDER}/openvla-7b+${DATASET_NAME}+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--${ID_NOTE}--${CKPT_NUMBER}_chkpt

# echo "*************************************************************"
# echo "Running evaluation for run ${RUN} with ID note: ${ID_NOTE} change_spawn_regions: ${CHANGE_SPAWN_REGIONS} chunk_size: ${CHUNK_SIZE} object_set: ${OBJECT_SET} change_command: ${CHANGE_COMMAND}"
# echo "*************************************************************"

# SAVE=True
# if [ "$RUN" -ne 1 ]; then
#     SAVE=False
# fi
# srun torchrun --standalone --nnodes 1 --nproc-per-node 1 run_robosuite_eval.py \
#     --task_suite_name ${TASK_SUITE_NAME} \
#     --wandb_entity "francescorosa97" \
#     --wandb_project "Open_VLA_OFT_finetune" \
#     --save ${SAVE} \
#     --run_number ${RUN} \
#     --change_spawn_regions ${CHANGE_SPAWN_REGIONS} \
#     --object_set ${OBJECT_SET} \
#     --change_command ${CHANGE_COMMAND} \

# ur5e_pick_place_delta_all 
# ur5e_pick_place_delta_removed_0_5_10_15
# ur5e_pick_place_removed_spawn_regions
# ur5e_pick_place_rm_one_spawn
# ur5e_pick_place_rm_central_spawn

RUN_ID=$1
CHANGE_SPAWN_REGIONS=$2
echo Running evaluation for run ${RUN_ID} with change_spawn_regions: ${CHANGE_SPAWN_REGIONS}
srun torchrun --standalone --nnodes 1 --nproc-per-node 1 run_robosuite_eval.py \
    --config_path="models/tinyvla_eval_config.yml" \
    --task_suite_name "ur5e_pick_place_delta_removed_0_5_10_15" \
    --run_number ${RUN_ID} \
    --change_spawn_regions ${CHANGE_SPAWN_REGIONS} \

