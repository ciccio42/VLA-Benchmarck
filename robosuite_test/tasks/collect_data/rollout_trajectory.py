from multiprocessing import Pool, cpu_count
from os.path import join
import torch
import json
import pickle as pkl
import functools
from multi_task_robosuite_env.controllers.controllers.expert_nut_assembly import \
    get_expert_trajectory as nut_expert
from multi_task_robosuite_env.controllers.controllers.expert_pick_place import \
    get_expert_trajectory as place_expert
from robosuite import load_controller_config
import pickle
import os
import cv2
import debugpy
import logging
from PIL import Image
import numpy as np
import random
import glob
from robosuite.utils import RandomizationError
from torchvision.transforms import ToTensor, Normalize
from torchvision.transforms.functional import resized_crop

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("Log")
logger.setLevel(logging.INFO)

# To be crystal clear: each constructed "Environment" is defined by both (task_name, robot_name), e.g. 'PandaBasketball'
# but each task_name may have differnt sub-task ids: e.g. Basketball-task_00 is throwing the white ball into hoop #1
TASK_ENV_MAP = {
    'pick_place': {
        'n_task':   16,
        'env_fn':   place_expert,
        'panda':    'Panda_PickPlaceDistractor',
        'sawyer':   'Sawyer_PickPlaceDistractor',
        'ur5e':     'UR5e_PickPlaceDistractor',
        'object_set': 2,
    },
    'nut_assembly':  {
        'n_task':   9,
        'env_fn':   nut_expert,
        'panda':    'Panda_NutAssemblyDistractor',
        'sawyer':   'Sawyer_NutAssemblyDistractor',
        'ur5e':     'UR5e_NutAssemblyDistractor',
    },
}

ROBOT_NAMES = ['panda', 'sawyer', 'ur5e']

NORMALIZATION_RANGES = np.array([[-0.35,  0.25],
                                [-0.30,  0.30],
                                 [0.60,  1.20],
                                 [-3.14,  3.14911766],
                                 [-3.14911766, 3.14911766],
                                 [-3.14911766,  3.14911766]])


def build_tvf_formatter():
    """Use this for torchvision.transforms in multi-task dataset, 
    note eval_fn always feeds in traj['obs']['images'], i.e. shape (h,w,3)
    """
    height = 100
    width = 180

    crop_params = [20, 25, 80, 75]
    # print(crop_params)
    top, left = crop_params[0], crop_params[2]

    def resize_crop(img):
        cv2.imwrite("obs.png", np.array(img))
        if len(img.shape) == 4:
            img = img[0]
        img_h, img_w = img.shape[0], img.shape[1]
        assert img_h != 3 and img_w != 3, img.shape
        box_h, box_w = img_h - top - \
            crop_params[1], img_w - left - crop_params[3]

        obs = ToTensor()(img[:, :, ::-1].copy())
        obs = resized_crop(obs, top=top, left=left, height=box_h, width=box_w,
                           size=(height, width), antialias=True)
        cv2.imwrite("cropped.png", np.moveaxis(obs.numpy()*255, 0, -1))
        obs = Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])(obs)

        return obs
    return resize_crop


def normalize_action(action, n_action_bin, action_ranges):
    half_action_bin = int(n_action_bin/2)
    norm_action = action.copy()
    # normalize between [-1 , 1]
    norm_action[:-1] = (2 * (norm_action[:-1] - action_ranges[:, 0]) /
                        (action_ranges[:, 1] - action_ranges[:, 0])) - 1
    # action discretization
    return (norm_action * half_action_bin).astype(np.int32).astype(np.float32) / half_action_bin


def denormalize_action(norm_action, action_ranges):
    action = np.clip(norm_action.copy(), -1, 1)
    for d in range(action_ranges.shape[0]):
        action[d] = (0.5 * (action[d] + 1) *
                     (action_ranges[d, 1] - action_ranges[d, 0])) + action_ranges[d, 0]
    return action


def init_env(env, traj, task_name):
    # get objects id
    if task_name == 'pick_place':
        for obj_name in env.object_to_id.keys():
            obj = env.objects[env.object_to_id[obj_name]]
            # set object position based on trajectory file
            obj_pos = traj[3]['obs'][f"{obj_name}_pos"]
            obj_quat = traj[3]['obs'][f"{obj_name}_quat"]
            env.sim.data.set_joint_qpos(
                obj.joints[0], np.concatenate([obj_pos, obj_quat]))
    elif task_name == 'nut_assembly':
        for obj_name in env.env.nut_to_id.keys():
            obj = env.env.nuts[env.env.nut_to_id[obj_name]]
            obj_id = env.env.nut_to_id[obj_name]
            if obj_id == 0:
                obj_pos = traj[1]['obs']['round-nut_pos']
                obj_quat = traj[1]['obs']['round-nut_quat']
            else:
                obj_pos = traj[1]['obs'][f'round-nut-{obj_id+1}_pos']
                obj_quat = traj[1]['obs'][f'round-nut-{obj_id+1}_quat']
            # set object position based on trajectory file
            env.sim.data.set_joint_qpos(
                obj.joints[0], np.concatenate([obj_pos, obj_quat]))


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_path', default="/", help="Path to task")
    parser.add_argument("--debug", action='store_true',
                        help="whether or not attach the debugger")
    parser.add_argument("--task_name", default='pick_place')
    args = parser.parse_args()

    if args.debug:
        debugpy.listen(('0.0.0.0', 5678))
        logger.info("Waiting for debugger attach")
        debugpy.wait_for_client()

    logger.info(f"Task path: {args.task_path}")
    task_paths = glob.glob(os.path.join(args.task_path, 'task_*'))

    img_formatter = build_tvf_formatter()

    # Load custom controller
    current_dir = os.path.dirname(os.path.abspath(__file__))
    controller_config_path = os.path.join(
        current_dir, "../multi_task_robosuite_env/controllers/config/osc_pose.json")
    print(f"Controller path {controller_config_path}")
    controller_config = load_controller_config(
        custom_fpath=controller_config_path)

    for dir in sorted(task_paths):
        print(dir)
        if os.path.isdir(os.path.join(args.task_path, dir)) and 'video' not in (os.path.join(args.task_path, dir)):
            # assert len(trjs) == 100, print(f"{os.path.join(args.task_path, dir)} does not have 100 trjs")
            trj_paths = glob.glob(os.path.join(dir, 'traj*.pkl'))

            for trj in sorted(trj_paths):
                print(trj)
                # logger.debug(f"Trajectory name: {dir}/{trj}")
                i = 0

                with open(trj, "rb") as f:
                    sample = pickle.load(f)
                    logger.debug(f"Sample keys {sample.keys()}")
                    logger.debug(sample)

                    # take the Trajectory obj from the trajectory
                    trajectory_obj = sample['traj']
                    i = 0
                    obj_in_hand = 0
                    start_moving = 0
                    end_moving = 0
                    # init env
                    env_fn = TASK_ENV_MAP[args.task_name]['env_fn']
                    env = env_fn('UR5e_PickPlaceDistractor',
                                 controller_type=controller_config,
                                 renderer=False,
                                 camera_obs=True,
                                 task=i,
                                 render_camera='camera_front',
                                 object_set=2,
                                 ret_env=True)

                    while True:
                        try:
                            obs = env.reset()
                            break
                        except RandomizationError:
                            pass
                    mj_state = env.sim.get_state().flatten()
                    sim_xml = env.model.get_xml()
                    env.reset_from_xml_string(sim_xml)
                    env.sim.reset()
                    env.sim.set_state_from_flattened(mj_state)
                    env.sim.forward()
                    # reset env according to current trajectory
                    init_env(env=env,
                             traj=trajectory_obj,
                             task_name=args.task_name)

                    for t in range(sample['len']):
                        # get action
                        step = trajectory_obj.get(t)
                        if t != 0:
                            action = step['action']
                            # env.render()
                            norm_action = normalize_action(
                                action=action, n_action_bin=256, action_ranges=NORMALIZATION_RANGES)
                            action = denormalize_action(
                                norm_action=norm_action, action_ranges=NORMALIZATION_RANGES)
                            obs, reward, env_done, info = env.step(action)
                            img_formatter(obs['camera_front_image'].copy())
                    del env
