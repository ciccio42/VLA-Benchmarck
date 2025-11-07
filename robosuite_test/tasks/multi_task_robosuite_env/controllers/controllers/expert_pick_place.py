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
import copy
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
pick_place_logger = logging.getLogger(name="PickPlaceLogger")

object_to_id = {"greenbox": 0, "yellowbox": 1, "bluebox": 2, "redbox": 3}

# in case rebuild is needed to use GPU render: sudo mkdir -p /usr/lib/nvidia-000
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
# pip uninstall mujoco_py; pip install mujoco_py


def _clip_delta(delta, max_step=0.015):
    norm_delta = np.linalg.norm(delta)
    if norm_delta < max_step:
        return delta
    return delta / norm_delta * max_step


class PickPlaceController:
    def __init__(self, env, ranges, tries=0, object_set=1):
        self._env = env
        self.ranges = ranges
        self._object_set = object_set
        self.reset()

    def _calculate_quat(self, obs):
        # Compute target quaternion that defines the final desired gripper orientation
        # 1. Obtain the orientation of the object wrt to world
        obj_quat = obs['{}_quat'.format(self._object_name)]
        if "nut" in self._object_name:
            obj_rot = T.quat2mat(obj_quat)
            obj_rot = np.array([[-1.0, 0.0, 0.0],
                                [0.0, -1.0, 0.0],
                                [0.0, 0.0, 1.0]])@obj_rot
        else:
            obj_rot = T.quat2mat(obj_quat)
        # 2. compute the new gripper orientation with respect to the gripper
        world_ee_rot = np.matmul(obj_rot, self._target_gripper_wrt_obj_rot)
        return Quaternion(matrix=world_ee_rot)

    def reset(self):
        self._object_name = self._env.objects[self._env.object_id].name
        # TODO this line violates abstraction barriers but so does the reference implementation in robosuite
        self._jpos_getter = lambda: np.array(self._env._joint_positions)
        self._clearance = 0.03  # 0.03 if 'milk' not in self._object_name else -0.01

        if "Sawyer" in self._env.robot_names:
            self._obs_name = 'eef_pos'
            self._default_speed = 0.02
            self._final_thresh = 1e-2
            # define the target gripper orientation with respect to the object
            self._target_gripper_wrt_obj_rot = np.array(
                [[1, 0, 0.], [0, -1, 0.], [0., 0., -1.]])
            self._g_tol = 1e-2
        elif "Panda" in self._env.robot_names:
            self._obs_name = 'eef_pos'
            self._default_speed = 0.02
            self._final_thresh = 6e-2
            # define the target gripper orientation with respect to the object
            self._target_gripper_wrt_obj_rot = np.array(
                [[0, 1, 0.], [1, 0, 0.], [0., 0., -1.]])
            self._g_tol = 5e-2
        elif "UR5e" in self._env.robot_names:
            self._obs_name = 'eef_pos'
            self._default_speed = 0.02
            self._final_thresh = 6e-2
            # define the target gripper orientation with respect to the object
            self._target_gripper_wrt_obj_rot = np.array(
                [[1, 0, 0.], [0, -1, 0.], [0., 0., -1.]])
            self._g_tol = 5e-2
        else:
            raise NotImplementedError

        # define the initial orientation of the gripper site
        self._base_quat = Quaternion(matrix=np.reshape(
            self._env.sim.data.site_xmat[self._env.robots[0].eef_site_id], (3, 3)))
        pick_place_logger.debug(
            f"Starting position:\n{self._env.sim.data.site_xpos[self._env.robots[0].eef_site_id]}")
        pick_place_logger.debug(
            f"Base rot:\n{np.reshape(self._env.sim.data.site_xmat[self._env.robots[0].eef_site_id], (3,3))}")

        self._t = 0
        self._intermediate_reached = False
        self._hover_delta = 0.20
        self._obj_thr = 0.10
        if self._object_set == 1:
            dist_panda = {'milk': 0.05, 'can': 0.018,
                          'cereal': 0.018, 'bread': 0.018}
            dist_sawyer = {'milk': 0.05, 'can': 0.018,
                           'cereal': 0.018, 'bread': 0.018}
            dist_ur5e = {'milk': 0.05, 'can': 0.03,
                         'cereal': 0.03, 'bread': 0.03}
            self.final_placing = [0, 0, 0.12]
        elif self._object_set == 2:
            dist_panda = {'greenbox': 0.05, 'yellowbox': 0.018,
                          'bluebox': 0.018, 'redbox': 0.018}
            dist_sawyer = {'greenbox': 0.05, 'yellowbox': 0.018,
                           'bluebox': 0.018, 'redbox': 0.018}
            dist_ur5e = {'greenbox': 0.05, 'yellowbox': 0.03,
                         'bluebox': 0.03, 'redbox': 0.03}
            self.final_placing = [0, 0, 0.12]
        elif self._object_set == 3:
            # {'greenbox': 0.05, 'yellowbox': 0.018,
            #  'bluebox': 0.018, 'redbox': 0.018}
            dist_panda = dict()
            # {'greenbox': 0.05, 'yellowbox': 0.018,
            #  'bluebox': 0.018, 'redbox': 0.018}
            dist_sawyer = dict()
            # {'greenbox': 0.05, 'yellowbox': 0.03,
            #  'bluebox': 0.03, 'redbox': 0.03}
            dist_ur5e = dict()
            for obj_name in self._env.obj_names:
                dist_panda[obj_name] = 0.03
                dist_sawyer[obj_name] = 0.03
                dist_ur5e[obj_name] = 0.03
            self.final_placing = [0, 0, 0.12]

        if "Panda" in self._env.robot_names:
            self.dist = dist_panda
            # gripper depth defines the distance between the TCP and the edge of the gripper
            self._gripper_depth = 0.01
        elif "Sawyer" in self._env.robot_names:
            self.dist = dist_sawyer
            # gripper depth defines the distance between the TCP and the edge of the gripper
            self._gripper_depth = 0.038/2
        elif "UR5e" in self._env.robot_names:
            self.dist = dist_ur5e
            # gripper depth defines the distance between the TCP and the edge of the gripper
            self._gripper_depth = 0.038/2

    def _object_in_hand(self, obs):
        # if np.linalg.norm(obs['{}_pos'.format(self._object_name)] - obs[self._obs_name]) < self.dist[self._object_name] \
        #    and (obs['{}_pos'.format(self._object_name)][2] - obs[self._obs_name][2]) > 0 \
        #    and (obs['{}_pos'.format(self._object_name)][2] - obs[self._obs_name][2]) <= self._gripper_depth:
        if np.linalg.norm(obs['{}_pos'.format(self._object_name)] - obs[self._obs_name]) < self.dist[self._object_name]:
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
        self._target_loc = np.array(
            self._env.sim.data.body_xpos[self._env.bin_bodyid]) + [0, 0, self._hover_delta]
        status = 'start'
        if self._t == 0:
            self._start_grasp = -1
            self._finish_grasp = False
            self._target_quat = self._calculate_quat(obs)
            self._move_up = False

        # Phase 1
        if self._start_grasp < 0:
            # check if the "prepare_grasp" phase is over
            if np.linalg.norm(obs['{}_pos'.format(self._object_name)][:2] - obs[self._obs_name][:2]) < self._g_tol:
                self._start_grasp = self._t

            # perform the inteporpolation between _base_quat and _target_quat
            quat_t = Quaternion.slerp(
                self._base_quat, self._target_quat, min(1, float(self._t) / 20))
            eef_pose = self._get_target_pose(
                obs['{}_pos'.format(self._object_name)] -
                obs[self._obs_name] + [0, 0, self._hover_delta],
                obs['eef_pos'], quat_t)
            action = np.concatenate((eef_pose, [-1]))
            status = 'prepare_grasp'
        # Phase 2
        elif self._start_grasp > 0 and not self._finish_grasp:
            if not self._object_in_hand(obs):
                # the object is not in the hand, approaching the object
                eef_pose = self._get_target_pose(
                    obs['{}_pos'.format(self._object_name)] -
                    obs[self._obs_name] - [0, 0, self._clearance],
                    obs['eef_pos'], self._target_quat)
                action = np.concatenate((eef_pose, [-1]))
                self.object_pos = obs['{}_pos'.format(self._object_name)]
                status = 'reaching_obj'
            else:
                # the object is in the hand, close the gripper and start the new phase
                eef_pose = self._get_target_pose(
                    self.object_pos - obs[self._obs_name], obs['eef_pos'], self._target_quat)
                action = np.concatenate((eef_pose, [1]))
                self._finish_grasp = True
                status = 'obj_in_hand'
        # Phase 3
        elif np.linalg.norm(
                self._target_loc - obs[self._obs_name]) > self._final_thresh and not self._intermediate_reached:
            if not self._move_up:
                self._init_obj_pos = obs['{}_pos'.format(self._object_name)]
                self._move_up = True
            # check the current object height
            if (abs(self._init_obj_pos[2] - obs['{}_pos'.format(self._object_name)][2]) < self._obj_thr):
                # target location is the current gripper position + security threshold
                target = obs['eef_pos'] + [0, 0, self._obj_thr]
            else:
                target = self._target_loc
            # moving towards the goal bin
            eef_pose = self._get_target_pose(
                target - obs[self._obs_name], obs['eef_pos'], self._target_quat)
            action = np.concatenate((eef_pose, [1]))
            status = 'moving'
        # Phase 4
        else:
            self._intermediate_reached = True
            if np.linalg.norm(self._target_loc - self.final_placing - obs[self._obs_name]) > self._final_thresh:
                target = self._target_loc - self.final_placing
                eef_pose = self._get_target_pose(
                    target - obs[self._obs_name], obs['eef_pos'], self._target_quat)
                action = np.concatenate((eef_pose, [1]))
            else:
                eef_pose = self._get_target_pose(
                    np.zeros(3), obs['eef_pos'], self._target_quat)
                action = np.concatenate((eef_pose, [-1]))
            status = 'placing'

        self._t += 1
        pick_place_logger.debug(f"Status {status}")
        return action, status


def get_expert_trajectory(env_type, controller_type, renderer=False, camera_obs=True, task=None, ret_env=False, seed=None, env_seed=None, gpu_id=0, render_camera="frontview", object_set=1, **kwargs):
    # assert 'gpu' in str(
    #     mujoco_py.cymj), 'Make sure to render with GPU to make eval faster'
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
            [[-0.05, 0.25], [-0.45, 0.5], [0.82, 1.2], [-5, 5], [-5, 5], [-5, 5]])
    elif 'UR5e' in env_type:
        action_ranges = np.array(
            [[-0.05, 0.25], [-0.45, 0.5], [0.82, 1.2], [-5, 5], [-5, 5], [-5, 5]])
    elif 'Panda' in env_type:
        action_ranges = np.array(
            [[-0.05, 0.25], [-0.45, 0.5], [0.82, 1.2], [0.85, 1.08], [-1, 1], [-1, 1], [-1, 1]])
    success, use_object = False, None
    if task is not None:
        assert 0 <= task <= 15, "task should be in [0, 15]"
    else:
        raise NotImplementedError

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
                          **kwargs)
            break
        except RandomizationError:
            pass
    while not success:
        controller = PickPlaceController(
            env.env, tries=tries, ranges=action_ranges, object_set=object_set)
        np.random.seed(seed + int(tries) + seed_offset)
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
        use_object = env.object_id
        traj.append(obs, raw_state=mj_state, info={'status': 'start'})
        print(f"Target object {controller._object_name}")
        for t in range(int(env.horizon/env.action_repeat)):
            # compute the action for the current state
            action, status = controller.act(obs)

            obs, reward, done, info = env.step(action)

            cv2.imwrite(f"debug.png", obs['camera_front_image'][:, :, ::-1])
            try:
                os.makedirs("test")
            except:
                pass
            image = np.array(obs['camera_front_image'][:, :, ::-1])
            for obj_name in obs['obj_bb']['camera_front']:
                obj_bb = obs['obj_bb']['camera_front'][obj_name]
                color = (0, 0, 255)
                image = cv2.rectangle(
                    image, (obj_bb['bottom_right_corner'][0], obj_bb['bottom_right_corner'][1]), (obj_bb['upper_left_corner'][0], obj_bb['upper_left_corner'][1]), color, 1)
            cv2.imwrite(f"test/prova.png",
                        image)
            assert 'status' not in info.keys(
            ), "Don't overwrite information returned from environment. "

            if renderer:
                env.render()

            mj_state = env.sim.get_state().flatten()
            traj.append(obs, reward, done, info, action, mj_state)
            # # plot bb
            # target_obj_id = obs['target-object']
            # target_obj_bb = None
            # for object_names in object_to_id.keys():
            #     if target_obj_id == object_to_id[object_names]:
            #         target_obj_bb = obs['obj_bb']['camera_front'][object_names]
            # image_rgb = np.array(obs['camera_front_image'][:, :, ::-1])
            # center = target_obj_bb['center']
            # upper_left_corner = target_obj_bb['upper_left_corner']
            # bottom_right_corner = target_obj_bb['bottom_right_corner']
            # image_rgb = cv2.circle(
            #     image_rgb, center, radius=1, color=(0, 0, 255), thickness=-1)
            # image_rgb = cv2.rectangle(
            #     image_rgb, upper_left_corner,
            #     bottom_right_corner, (255, 0, 0), 1)
            # cv2.imshow('camera_front_image', image_rgb)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
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
    import yaml
    debugpy.listen(('0.0.0.0', 5678))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
    # Load configuration files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    controller_config_path = os.path.join(
        current_dir, "../config/osc_pose.json")
    controller_config = load_controller_config(
        custom_fpath=controller_config_path)

    for i in range(0, 16):
        traj = get_expert_trajectory('UR5e_PickPlaceDistractor',
                                     controller_type=controller_config,
                                     renderer=False,
                                     camera_obs=True,
                                     task=i,
                                     render_camera='camera_front',
                                     object_set=2)
