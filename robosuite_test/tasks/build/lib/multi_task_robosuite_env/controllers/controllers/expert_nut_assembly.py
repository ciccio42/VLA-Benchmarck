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
nut_assembly_logger = logging.getLogger(name="NutAssemblyLogger")


def _clip_delta(delta, max_step=0.015):
    norm_delta = np.linalg.norm(delta)
    if norm_delta < max_step:
        return delta
    return delta / norm_delta * max_step


class NutAssemblyController:
    def __init__(self, env, ranges, tries=0):
        assert env.single_object_mode == 2, "only supports single object environments at this point!"
        self._env = env
        self._g_tol = 5e-2 ** (tries + 1)
        self.reset()
        self.ranges = ranges

    def _calculate_quat(self, obs):
        # Compute target quaternion that defines the final desired gripper orientation
        # 1. Obtain the orientation of the object wrt to world
        obj_quat = obs['{}_quat'.format(self._object_name)]
        obj_rot = T.quat2mat(obj_quat)
        # 2. compute the new gripper orientation with respect to the gripper
        world_ee_rot = np.matmul(obj_rot, self._target_gripper_wrt_obj_rot)
        return Quaternion(matrix=world_ee_rot)

    def get_handle_loc(self):
        if self._env.nut_id == 0:
            handle_loc = self._env.sim.data.site_xpos[self._env.sim.model.site_name2id(
                'round-nut_handle_site')]
        elif self._env.nut_id == 1:
            handle_loc = self._env.sim.data.site_xpos[self._env.sim.model.site_name2id(
                'round-nut-2_handle_site')]
        else:
            handle_loc = self._env.sim.data.site_xpos[self._env.sim.model.site_name2id(
                'round-nut-3_handle_site')]
        handle_loc[2] = max(handle_loc[2], 0.01)
        return handle_loc

    def get_center_loc(self):
        if self._env.nut_id == 0:
            center_loc = self._env.sim.data.site_xpos[self._env.sim.model.site_name2id(
                'round-nut_center_site')]
        elif self._env.nut_id == 1:
            center_loc = self._env.sim.data.site_xpos[self._env.sim.model.site_name2id(
                'round-nut-2_center_site')]
        else:
            center_loc = self._env.sim.data.site_xpos[self._env.sim.model.site_name2id(
                'round-nut-3_center_site')]
        center_loc[2] = max(center_loc[2], 0.005)
        return center_loc

    def reset(self):
        self._object_name = self._env.nuts[self._env.nut_id].name
        if self._env.default_peg == 0:
            self._target_loc = np.array(
                self._env.sim.data.body_xpos[self._env.peg1_body_id]) + [0, 0, 0.2]
            self._target_peg_base = np.array(
                self._env.sim.data.body_xpos[self._env.peg1_body_id])
        elif self._env.default_peg == 1:
            self._target_loc = np.array(
                self._env.sim.data.body_xpos[self._env.peg2_body_id]) + [0, 0, 0.2]
            self._target_peg_base = np.array(
                self._env.sim.data.body_xpos[self._env.peg2_body_id])
        else:
            self._target_loc = np.array(
                self._env.sim.data.body_xpos[self._env.peg3_body_id]) + [0, 0, 0.2]
            self._target_peg_base = np.array(
                self._env.sim.data.body_xpos[self._env.peg3_body_id])
        self._clearance = 0.05

        if "Sawyer" in self._env.robot_names:
            self._obs_name = 'eef_pos'
            self._default_speed = 0.01
            self._final_thresh = 1e-2
            # define the target gripper orientation with respect to the object
            self._target_gripper_wrt_obj_rot = np.array(
                [[1, 0, 0.], [0, -1, 0.], [0., 0., -1.]])
        elif "Panda" in self._env.robot_names:
            self._obs_name = 'eef_pos'
            self._default_speed = 0.02
            self._final_thresh = 1e-2
            # define the target gripper orientation with respect to the object
            self._target_gripper_wrt_obj_rot = np.array(
                [[0, -1, 0.], [-1, 0, 0.], [0., 0., -1.]])
        elif "UR5e" in self._env.robot_names:
            self._obs_name = 'eef_pos'
            self._default_speed = 0.02
            self._final_thresh = 1e-2
            # define the target gripper orientation with respect to the object
            self._target_gripper_wrt_obj_rot = np.array(
                [[-1, 0, 0.], [0, 1, 0.], [0., 0., -1.]])
        else:
            raise NotImplementedError

        # define the initial orientation of the gripper site
        self._base_quat = Quaternion(matrix=np.reshape(
            self._env.sim.data.site_xmat[self._env.robots[0].eef_site_id], (3, 3)))
        nut_assembly_logger.debug(
            f"Starting position:\n{self._env.sim.data.site_xpos[self._env.robots[0].eef_site_id]}")
        nut_assembly_logger.debug(
            f"Base rot:\n{np.reshape(self._env.sim.data.site_xmat[self._env.robots[0].eef_site_id], (3,3))}")

        self._t = 0
        self._intermediate_reached = False
        self._hover_delta = 0.15

        if "Panda" in self._env.robot_names:
            # gripper depth defines the distance between the TCP and the edge of the gripper
            self._gripper_depth = 0.01
        elif "Sawyer" in self._env.robot_names:
            # gripper depth defines the distance between the TCP and the edge of the gripper
            self._gripper_depth = 0.038/2
        elif "UR5e" in self._env.robot_names:
            # gripper depth defines the distance between the TCP and the edge of the gripper
            self._gripper_depth = 0.038/2

    def _object_in_hand(self, obs):
        if (self.get_handle_loc()[2] - obs[self._obs_name][2] > 0) and (self.get_handle_loc()[2] - obs[self._obs_name][2] <= self._gripper_depth):
            return True
        elif self._env._check_grasp(gripper=self._env.robots[0].gripper, object_geoms=[g for g in self._env.nuts[self._env.nut_id].contact_geoms]):
            return True
        return False

    def _get_target_pose(self, delta_pos, base_pos, quat, max_step=None):
        if max_step is None:
            max_step = self._default_speed

        delta_pos = _clip_delta(delta_pos, max_step)

        delta_pos = _clip_delta(delta_pos, max_step)
        quat = np.array([quat.x, quat.y, quat.z, quat.w])
        aa = quat2axisangle(quat)

        # absolute in world frame
        return np.concatenate((delta_pos + base_pos, aa))

    def act(self, obs, prev_status=None):
        status = 'start'
        if self._t == 0:
            self._start_grasp = -1
            self._finish_grasp = False
            self._target_quat = self._calculate_quat(obs)

        # Phase 1
        if self._start_grasp < 0:
            if (np.linalg.norm(self.get_handle_loc()[:2] - obs[self._obs_name][:2])) < self._g_tol:
                self._start_grasp = self._t

            quat_t = Quaternion.slerp(
                self._base_quat, self._target_quat, min(1, float(self._t) / 5))
            eef_pose = self._get_target_pose(
                self.get_handle_loc() -
                obs[self._obs_name] + [0, 0, self._hover_delta],
                obs['eef_pos'], quat_t)
            action = np.concatenate((eef_pose, [-1]))
            status = 'prepare_grasp'
        # Phase 2
        elif self._start_grasp >= 0 and not self._finish_grasp:
            if not self._object_in_hand(obs):
                # the object is not in the hand, approaching the object
                eef_pose = self._get_target_pose(
                    self.get_handle_loc() -
                    obs[self._obs_name] - [0, 0, self._clearance],
                    obs['eef_pos'], self._target_quat)
                action = np.concatenate((eef_pose, [-1]))
                self.object_pos = obs['{}_pos'.format(self._object_name)]
                status = 'reaching_obj'
            else:
                # the object is in the hand, close the gripper and start the new phase
                eef_pose = self._get_target_pose(self.get_handle_loc(
                ) - obs[self._obs_name], obs['eef_pos'], self._target_quat)
                action = np.concatenate((eef_pose, [1]))
                # if np.linalg.norm(self.object_pos - obs[self._obs_name] + [0, 0, self._hover_delta]) < self._g_tol:
                self._finish_grasp = True
                status = 'obj_in_hand'
        # Phase 3
        elif np.linalg.norm(
                self._target_loc - self.get_center_loc()) > self._final_thresh and prev_status != "assembling":
            target = self._target_loc
            eef_pose = self._get_target_pose(
                target - self.get_center_loc(), obs['eef_pos'], self._target_quat)
            action = np.concatenate((eef_pose, [1]))
            status = 'moving'
        # Phase 4
        # check whether the nut is inserted
        elif (self.get_center_loc()[2]-self._target_peg_base[2]) > 0.02:
            eef_pose = self._get_target_pose([0, 0, self._target_peg_base[2]-obs[self._obs_name][2]],
                                             obs['eef_pos'], self._target_quat)
            action = np.concatenate((eef_pose, [1]))
            status = 'assembling'
        else:
            if prev_status == 'assembling':
                eef_pose = self._get_target_pose(
                    0, obs['eef_pos'], self._target_quat)
                action = np.concatenate((eef_pose, [-1]))
                status = 'releasing'
        self._t += 1
        nut_assembly_logger.debug(f"Status {status}")
        return action, status

    def disconnect(self):
        p.disconnect()


def get_expert_trajectory(env_type, controller_type, renderer=False, camera_obs=True, task=None, ret_env=False, seed=None, env_seed=None, gpu_id=0, render_camera="frontview",  object_set=1, **kwargs):
    # assert widths == 180, widths
    assert 'gpu' in str(
        mujoco_py.cymj), 'Make sure to render with GPU to make eval faster'
    # reassign the gpu id
    visible_ids = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if gpu_id == 3:
        gpu_id = 0
    elif gpu_id == 0:
        gpu_id = 3
    elif gpu_id == 1:
        gpu_id = 2
    elif gpu_id == 2:
        gpu_id = 1

    seed = seed if seed is not None else random.getrandbits(32)
    env_seed = seed if env_seed is None else env_seed
    seed_offset = sum([int(a) for a in bytes(env_type, 'ascii')])
    np.random.seed(env_seed)

    if 'Sawyer' in env_type:
        action_ranges = np.array(
            [[-0.05, 0.25], [-0.45, 0.5], [0.82, 1.2], [-5, 5], [-5, 5], [-5, 5]])
    elif 'UR5e' in env_type:
        action_ranges = np.array(
            [[-0.05, 0.25], [-0.45, 0.5], [0.82, 1.2], [-5, 5], [-5, 5], [-5, 5]])
    elif 'Panda' in env_type:
        action_ranges = np.array(
            [[-0.05, 0.25], [-0.45, 0.5], [0.82, 1.2], [0.85, 1.08], [-1, 1], [-1, 1], [-1, 1]])

    success, use_object = False, None
    if task is not None:
        assert 0 <= task <= 8, "task should be in [0, 8]"
        use_object = int(task // 3)
        peg_id = task % 3

    if ret_env:
        while True:
            try:
                env = get_env(env_type,
                              force_object=use_object,
                              default_peg=peg_id,
                              controller_configs=controller_type,
                              has_renderer=renderer,
                              has_offscreen_renderer=camera_obs,
                              reward_shaping=False,
                              use_camera_obs=camera_obs,
                              ranges=action_ranges,
                              render_camera=render_camera,
                              render_gpu_device_id=gpu_id,
                              **kwargs)
                break
            except RandomizationError:
                pass
        return env

    tries = 0
    while True:
        try:
            env = get_env(env_type,
                          force_object=use_object,
                          default_peg=peg_id,
                          controller_configs=controller_type,
                          has_renderer=renderer,
                          has_offscreen_renderer=camera_obs,
                          reward_shaping=False,
                          use_camera_obs=camera_obs,
                          ranges=action_ranges,
                          render_camera=render_camera,
                          render_gpu_device_id=gpu_id,
                          **kwargs)
            break
        except RandomizationError:
            pass
    while not success:
        controller = NutAssemblyController(
            env.env, tries=tries, ranges=action_ranges)
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
        status = "start"
        for t in range(int(env.horizon/env.action_repeat)):
            action, status = controller.act(obs, status)
            obs, reward, done, info = env.step(action)
            cv2.imwrite(f"debug.png", obs['camera_front_image'][:, :, ::-1])
            assert 'status' not in info.keys(
            ), "Don't overwrite information returned from environment. "
            info['status'] = status
            if renderer:
                env.render()
            mj_state = env.sim.get_state().flatten()
            traj.append(obs, reward, done, info, action, mj_state)
            # print(done)
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
    debugpy.listen(('0.0.0.0', 5678))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()

    # Load configuration files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    controller_config_path = os.path.join(
        current_dir, "../config/osc_pose.json")
    controller_config = load_controller_config(
        custom_fpath=controller_config_path)

    for i in range(9):
        traj = get_expert_trajectory('Panda_NutAssemblyDistractor',
                                     controller_type=controller_config,
                                     renderer=False,
                                     camera_obs=True,
                                     task=i,
                                     render_camera='camera_front')
