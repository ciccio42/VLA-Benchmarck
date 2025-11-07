import sys
from pathlib import Path

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))
import numpy as np
from multi_task_il.datasets import Trajectory
try:
    import pybullet as p
except:
    pass
from pyquaternion import Quaternion
import random
from multi_task_robosuite_env.custom_ik_wrapper import normalize_action
from robosuite import load_controller_config
from robosuite.utils.transform_utils import quat2axisangle
from robosuite.utils import RandomizationError
import torch
import os
# import mujoco_py
import robosuite.utils.transform_utils as T
import multi_task_robosuite_env.utils as utils
from multi_task_robosuite_env import get_env
import cv2
# in case rebuild is needed to use GPU render: sudo mkdir -p /usr/lib/nvidia-000
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
# pip uninstall mujoco_py; pip install mujoco_py

import copy
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
button_logger = logging.getLogger(name="ButtonLogger")


def _clip_delta(delta, max_step=0.015):
    norm_delta = np.linalg.norm(delta)
    if norm_delta < max_step:
        return delta
    return delta / norm_delta * max_step


class ButtonPressController:
    def __init__(self, env, ranges, tries=0, object_set=1):
        self._env = env
        self.ranges = ranges
        self._object_set = object_set
        self._status = None
        self.reset()

    def _calculate_quat(self):
        # Compute target quaternion that defines the final desired gripper orientation
        # 1. Obtain the orientation of the object wrt to world
        obj_quat = T.mat2quat(
            np.reshape(np.array(self._env.sim.data.site_xmat[self._env.target_button_id]), (3, 3)))
        obj_rot = T.quat2mat(obj_quat)
        # 2. compute the new gripper orientation with respect to the gripper
        world_ee_rot = np.matmul(obj_rot, self._target_gripper_wrt_obj_rot)
        return Quaternion(matrix=world_ee_rot)

    def get_button_loc(self):
        button_loc = np.array(
            self._env.sim.data.site_xpos[self._env.target_button_id])

        # button quat
        # button_mat = np.array(
        #     self._env.sim.data.site_xmat[self._env.target_button_id])
        # button_quat = T.mat2quat(button_mat)

        return button_loc

    def get_dir(self):
        dir = [np.array([0, 1., 0]), np.array([0., -1., 0.])]
        return dir[self._env.task_id // 3] * 0.2

    def reset(self):

        self._clearance = 0.03  # 0.03 if 'milk' not in self._object_name else -0.01

        if "Sawyer" in self._env.robot_names:
            self._obs_name = 'eef_pos'
            self._default_speed = 0.02
            self._final_thresh = 1e-3
            # define the target gripper orientation with respect to the object
            self._target_gripper_wrt_obj_rot = np.array(
                [[1, 0, 0.], [0, -1, 0.], [0., 0., -1.]])
            self._g_tol = 1e-2
        elif "Panda" in self._env.robot_names:
            self._obs_name = 'eef_pos'
            self._default_speed = 0.02
            self._final_thresh = 1e-2
            # define the target gripper orientation with respect to the object
            self._target_gripper_wrt_obj_rot = np.array(
                [[1, 0, 0.], [0, -1, 0.], [0., 0., -1.]])
            self._g_tol = 5e-2
        elif "UR5e" in self._env.robot_names:
            self._obs_name = 'eef_pos'
            self._default_speed = 0.02
            self._final_thresh = 1e-2
            # define the target gripper orientation with respect to the object
            self._target_gripper_wrt_obj_rot = np.array(
                [[0, -1, 0.], [-1, 0, 0.], [0., 0., -1.]])
            self._g_tol = 5e-2
        else:
            raise NotImplementedError

        # define the initial orientation of the gripper site
        self._base_quat = Quaternion(matrix=np.reshape(
            self._env.sim.data.site_xmat[self._env.robots[0].eef_site_id], (3, 3)))
        # print(
        #     f"Starting position:\n{self._env.sim.data.site_xpos[self._env.robots[0].eef_site_id]}")
        # print(
        #     f"Base rot:\n{np.reshape(self._env.sim.data.site_xmat[self._env.robots[0].eef_site_id], (3,3))}")

        self._t = 0
        self._intermediate_reached = False
        self._hover_delta = 0.15

    def _reached_button(self, obs):
        # print(np.linalg.norm(self.get_button_loc() - obs[self._obs_name]))
        if "Sawyer" in self._env.robot_names:
            dist = 0.018
        else:
            dist = 0.030
        if np.linalg.norm(self.get_button_loc() - obs[self._obs_name]) < dist:
            return True
        return False

    def _get_target_pose(self, delta_pos, base_pos, quat, max_step=None):
        if max_step is None:
            max_step = self._default_speed

        delta_pos = _clip_delta(delta_pos, max_step)
        quat = np.array([quat.x, quat.y, quat.z, quat.w])
        aa = quat2axisangle(quat)

        # absolute in world frame
        return np.concatenate((delta_pos + base_pos, aa))

    def act(self, obs):
        status = 'start'
        if self._t == 0:
            self._start_reach = -1
            self._finish_reach = False

            # self._target_loc = np.array(
            #     self.get_button_loc() + [0, 0, self._hover_delta])
            # self._target_quat = self._calculate_quat(
            #     obs)

            # Original
            self._target_quat = self._calculate_quat()

        if self._start_reach < 0 and self._t < 15:
            if np.linalg.norm(self.get_button_loc() - obs[self._obs_name] + [0, 0, self._hover_delta]) < self._g_tol or self._t == 14:
                self._start_reach = self._t

            quat_t = Quaternion.slerp(
                self._base_quat, self._target_quat, min(1, float(self._t) / 5))
            eef_pose = self._get_target_pose(
                self.get_button_loc() -
                obs[self._obs_name] + [0, 0, self._hover_delta],
                obs['eef_pos'], quat_t)
            action = np.concatenate((eef_pose, [-1]))
            status = 'prepare_button'

        elif not self._reached_button(obs) and self._t < 35:
            eef_pose = self._get_target_pose(
                self.get_button_loc() - obs[self._obs_name],
                obs['eef_pos'], self._target_quat)
            action = np.concatenate((eef_pose, [-1]))
            status = 'reaching_button'
        else:
            eef_pose = self._get_target_pose(
                self.get_dir(), obs['eef_pos'], self._target_quat)
            action = np.concatenate((eef_pose, [1]))
            # eef_pose = self._get_target_pose(
            #     delta,
            #     obs['eef_pos'], self._target_quat)
            action = np.concatenate((eef_pose, [1]))
            status = 'press_button'

        self._t += 1
        self._status = status
        # print(f"Status {status}")
        return action, status


def get_expert_trajectory(env_type, controller_type, renderer=False, camera_obs=True, task=None, ret_env=False, seed=None, env_seed=None, render_camera="frontview", gpu_id=0, object_set=1, **kwargs):

    assert 'gpu' in str(mujoco_py.cymj), 'Make sure to render with GPU.'
    # reassign the gpu id
    visible_ids = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if gpu_id == 3:
        gpu_id = 0
    elif gpu_id == 0:
        gpu_id = 3  # 1
    elif gpu_id == 1:
        gpu_id = 2
    elif gpu_id == 2:
        gpu_id = 1  # 3
    print(f"GPU-ID {gpu_id}")
    seed = seed if seed is not None else random.getrandbits(32)
    env_seed = seed if env_seed is None else env_seed
    seed_offset = sum([int(a) for a in bytes(env_type, 'ascii')])
    np.random.seed(env_seed)
    if 'Sawyer' in env_type:
        action_ranges = np.array(
            [[-0.3, 0.3], [-0.3, 0.3], [0.78, 1.2], [-5, 5], [-5, 5], [-5, 5]])
    elif 'UR5e' in env_type:
        action_ranges = np.array(
            [[-0.05, 0.25], [-0.45, 0.5], [0.82, 1.2], [-5, 5], [-5, 5], [-5, 5]])
    elif 'Panda' in env_type:
        action_ranges = np.array(
            [[-0.3, 0.3], [-0.3, 0.3], [0.78, 1.2], [-1, 1], [-1, 1], [-1, 1], [-1, 1]])

    success, use_object = False, None
    if task is not None:
        assert 0 <= task <= 5, "task should be in [0, 5]"
        use_object = int(task)

    if ret_env:
        while True:
            try:
                env = get_env(env_type,
                              controller_configs=controller_type,
                              task_id=task,
                              has_renderer=renderer,
                              has_offscreen_renderer=camera_obs,
                              reward_shaping=False,
                              use_camera_obs=camera_obs,
                              ranges=action_ranges,
                              render_gpu_device_id=gpu_id,
                              render_camera=render_camera,
                              object_set=object_set,
                              ** kwargs)
                break
            except RandomizationError:
                pass
        return env

    tries = 0
    while True:
        try:
            env = get_env(env_type,
                          controller_configs=controller_type,
                          task_id=task,
                          has_renderer=renderer,
                          has_offscreen_renderer=camera_obs,
                          reward_shaping=False,
                          use_camera_obs=camera_obs,
                          ranges=action_ranges,
                          render_gpu_device_id=gpu_id,
                          render_camera=render_camera,
                          object_set=object_set,
                          ** kwargs)

            break
        except RandomizationError:
            pass
    while not success:
        controller = ButtonPressController(
            env.env,
            tries=tries,
            ranges=action_ranges)
        np.random.seed(seed + int(tries // 3) + seed_offset)
        while True:
            try:
                obs = env.reset()
                break
            except RandomizationError:
                pass
        mj_state = env.sim.get_state().flatten()
        sim_xml = env.model.get_xml()
        traj = Trajectory(sim_xml)

        env.reset_from_xml_string(sim_xml)
        env.sim.reset()
        env.sim.set_state_from_flattened(mj_state)
        env.sim.forward()
        traj.append(obs, raw_state=mj_state, info={'status': 'start'})
        cv2.imwrite(
            "env_initialization.png", obs['camera_front_image'][:, :, ::-1])
        for t in range(int(env.horizon // 10)):
            action, status = controller.act(obs)
            obs, reward, done, info = env.step(action)
            cv2.imwrite(
                "debug.png", obs['camera_front_image'][:, :, ::-1])
            assert 'status' not in info.keys(
            ), "Don't overwrite information returned from environment. "
            info['status'] = status
            if renderer:
                env.render()
            mj_state = env.sim.get_state().flatten()
            traj.append(obs, reward, done, info, action, mj_state)

            if reward:
                success = True
                break
        tries += 1

    if renderer:
        env.close()
    del controller
    del env
    return traj


if __name__ == '__main__':
    import debugpy
    import os
    import sys
    debugpy.listen(('0.0.0.0', 5678))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
    # Load configuration files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    controller_config_path = os.path.join(
        current_dir, "../config/osc_pose.json")
    controller_config = load_controller_config(
        custom_fpath=controller_config_path)
    for i in range(3, 4):
        traj = get_expert_trajectory('UR5e_Button',
                                     controller_type=controller_config,
                                     renderer=False,
                                     camera_obs=True,
                                     task=i,
                                     render_camera='camera_front',
                                     object_set=1)
