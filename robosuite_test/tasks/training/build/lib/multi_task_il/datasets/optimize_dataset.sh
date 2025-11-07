#!/bin/bash

DATASET_PATH=/user/frosa/multi_task_lfd/ur_multitask_dataset
TASK_NAME=press_button_close_after_reaching
OUT_PATH=/raid/home/frosa_Loc/opt_dataset/

python optimize_dataset.py --dataset_path ${DATASET_PATH} --task_name ${TASK_NAME} --robot_name ur5e --out_path ${OUT_PATH}

python optimize_dataset.py --dataset_path ${DATASET_PATH} --task_name ${TASK_NAME} --robot_name panda --out_path ${OUT_PATH}
