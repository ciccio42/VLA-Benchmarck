#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

MODEL=/home/frosa_loc/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/ur-baseline/1Task-Pick-Place-Mosaic-No-Obj-Detector-Ur5e-Batch128
EXP_NUMBER=1
STEP=102212
TASK_ID=-1 #2

RESULTS_DIR=/home/frosa_loc/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/ur-baseline/1Task-Pick-Place-Mosaic-No-Obj-Detector-Ur5e-Batch128/training_results
NUM_WORKERS=1
# PROJECT_NAME="ur_mosaic_baseline_no_obj_detector_training_rollout"

python dataset_analysis_ur.py --step ${STEP} --model ${MODEL} --task_indx ${TASK_ID} --results_dir ${RESULTS_DIR} --num_workers ${NUM_WORKERS} --experiment_number ${EXP_NUMBER} --training_trj --debug # --project_name ${PROJECT_NAME} --debug
