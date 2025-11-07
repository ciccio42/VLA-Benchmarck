"""\
This file loads the trajectories in pkl format from the specified folder and add the bounding-box related to the objects in the scene
The bounding box is defined as follow: (center_x, center_y, width, height)
"""

from multi_task_il.datasets.savers import _compress_obs
import os
import sys
import pickle
import cv2
import numpy as np
import logging
import copy
import robosuite.utils.transform_utils as T
import functools
from multiprocessing import Pool, cpu_count
import glob
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import resized_crop
import yaml

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger("BB-Creator")


def overwrite_pkl_file(pkl_file_path, sample, traj_obj_bb):
    # get trajectory from sample
    new_sample = copy.deepcopy(sample)

    traj = new_sample['traj']

    # modify trajectory observation
    for t in range(len(traj)):
        try:
            obs = traj.get(t)['obs']
        except:
            _img = traj._data[t][0]['camera_front_image']
            okay, im_string = cv2.imencode(
                '.jpg', _img)
            traj._data[t][0]['camera_front_image'] = im_string
            obs = traj.get(t)['obs']

        obs['obj_bb'] = traj_obj_bb[t]
        obs = _compress_obs(obs)
        traj.change_obs(t, obs)
        logger.debug(obs.keys())

    pickle.dump({
        'traj': traj,
        'len': len(traj),
        'env_type': sample['env_type'],
        'task_id': sample['task_id']}, open(pkl_file_path, 'wb'))


def prior_crop_resize(task_name, task_spec, pkl_file_path,):
    # pkl_file_path = os.path.join(task_path, pkl_file_path)
    # logger.info(f"Task id {dir} - Trajectory {pkl_file_path}")
    # 2. Load pickle file
    with open(pkl_file_path, "rb") as f:
        sample = pickle.load(f)
    # 3. Identify objects positions
    traj = sample['traj']
    traj_bb = []
    for t in range(len(sample['traj'])):

        for camera_name in ["camera_front"]:
            # take current image
            img = traj[t].get('obs').get(f'{camera_name}_img')[:, :, ::-1]
            cv2.imwrite("original.png", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            img_height, img_width = obs.shape[:2]
            """applies to every timestep's RGB obs['camera_front_image']"""
            crop_params = task_spec.get(
                task_name, [0, 0, 0, 0])
            top, left = crop_params[0], crop_params[2]
            img_height, img_width = obs.shape[0], obs.shape[1]
            box_h, box_w = img_height - top - \
                crop_params[1], img_width - left - crop_params[3]

            obs = ToTensor(obs)
            # ---- Resized crop ----#
            # start = time.time()
            obs = resized_crop(obs, top=top, left=left, height=box_h,
                               width=box_w, size=(100, 180))
            cv2.imwrite("cropped.png", obs)


if __name__ == '__main__':
    import debugpy
    # debugpy.listen(('0.0.0.0', 5678))
    # print("Waiting for debugger attach")
    # debugpy.wait_for_client()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_path', default="/", help="Path to task")
    parser.add_argument('--task_name', default="/", help="Name of the task")
    parser.add_argument('--robot_name', default="/", help="Name of the robot")
    args = parser.parse_args()

    # 1. Load the dataset
    folder_path = os.path.join(
        args.task_path, args.task_name, f"{args.robot_name}_{args.task_name}")

    # load task configuration file
    conf_file_path = "../../experiments/tasks_cfgs/7_tasks.yaml"
    with open(conf_file_path, 'r') as file:
        task_conf = yaml.safe_load(file)

    for dir in os.listdir(folder_path):
        print(dir)
        if "task_" in dir:
            task_path = os.path.join(folder_path, dir)
            print(task_path)
            i = 0
            trj_list = glob.glob(f"{task_path}/*.pkl")
            with Pool(1) as p:
                f = functools.partial(prior_crop_resize,
                                      args.task_name,
                                      task_conf)
                p.map(f, trj_list)
