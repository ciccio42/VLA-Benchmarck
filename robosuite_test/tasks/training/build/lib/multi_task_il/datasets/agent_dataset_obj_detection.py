import random
import torch
from multi_task_il.datasets import load_traj
import cv2
from torch.utils.data import Dataset


import pickle as pkl
from collections import defaultdict, OrderedDict
import glob
import numpy as np
import matplotlib.pyplot as plt
import copy

from multi_task_il.utils import normalize_action
from multi_task_il.datasets.utils import *


class AgentDatasetObjDetection(Dataset):
    def __init__(
            self,
            tasks_spec,
            root_dir='mosaic_multitask_dataset',
            mode='train',
            split=[0.9, 0.1],
            obs_T=7,
            bbs_T=1,
            action_T=1,
            take_first_frame=False,
            aug_twice=True,
            width=180,
            height=100,
            data_augs=None,
            non_sequential=False,
            state_spec=('ee_aa', 'gripper_qpos'),
            action_spec=('action',),
            allow_val_skip=False,
            allow_train_skip=False,
            use_strong_augs=False,
            aux_pose=False,
            select_random_frames=True,
            balance_target_obj_pos=True,
            compute_obj_distribution=False,
            agent_name='ur5e',
            normalize_action=True,
            pick_next=False,
            normalization_ranges=[],
            n_action_bin=256,
            perform_augs=True,
            perform_scale_resize=True,
            change_command_epoch=True,
            load_eef_point=False,
            ** params):

        self.task_crops = OrderedDict()
        self.agent_crop = OrderedDict()
        # each idx i maps to a unique tuple of (task_name, sub_task_id, agent.pkl, demo.pkl)
        self.all_file_pairs = OrderedDict()
        self.all_agent_files = OrderedDict()
        self.task_to_idx = defaultdict(list)
        self.subtask_to_idx = OrderedDict()
        self.agent_files = dict()
        self.agent_name = agent_name
        self.mode = mode
        self.pick_next = pick_next

        self.select_random_frames = select_random_frames
        self.balance_target_obj_pos = balance_target_obj_pos
        self.compute_obj_distribution = compute_obj_distribution
        if "real" in agent_name:
            self.real = True

        else:
            self.real = False

        self.object_distribution = OrderedDict()
        self.object_distribution_to_indx = OrderedDict()

        self._take_first_frame = take_first_frame

        self._selected_target_frame_distribution_task_object_target_position = OrderedDict()

        self._compute_frame_distribution = False
        self._normalize_action = normalize_action
        self._normalization_ranges = np.array(normalization_ranges)
        self._n_action_bin = n_action_bin
        self._perform_augs = perform_augs
        self._state_spec = state_spec
        self._load_state_spec = True if state_spec is not None else False
        self._change_command_epoch = change_command_epoch
        self._load_eef_point = load_eef_point
        self.perform_scale_resize = perform_scale_resize

        # Frame distribution for each trajectory
        self._frame_distribution = OrderedDict()
        self._mix_demo_agent = False
        count = create_train_val_dict(self,
                                      agent_name,
                                      root_dir,
                                      tasks_spec,
                                      split,
                                      allow_train_skip,
                                      allow_val_skip)

        if not self._change_command_epoch:
            self.pairs_count = count
        else:
            self.file_count = count

        self.task_count = len(tasks_spec)

        self._obs_T, self._bbs_T,  self._action_T = obs_T, bbs_T, action_T
        self.width, self.height = width, height
        self.aug_twice = aug_twice
        self.aux_pose = aux_pose

        self._state_action_spec = (state_spec, action_spec)
        self.non_sequential = non_sequential
        if non_sequential:
            print("Warning! The agent observations are not sampled in neighboring timesteps, make sure inverse dynamics loss is NOT used in training \n ")

        self.use_strong_augs = use_strong_augs
        self.data_augs = data_augs
        self.frame_aug = create_data_aug(self)

    def __len__(self):
        """NOTE: we should count total possible demo-agent pairs, not just single-file counts
        total pairs should sum over all possible sub-task pairs"""
        if not self._change_command_epoch:
            return self.pairs_count
        else:
            return self.file_count

    def __getitem__(self, idx):
        """since the data is organized by task, use a mapping here to convert
        an index to a proper sub-task index """
        if self.mode == 'train':
            pass
        if not self._change_command_epoch:
            (task_name, sub_task_id, demo_file,
             agent_file) = self.all_file_pairs[idx]
        else:
            (task_name, sub_task_id, agent_file,
                trj_length) = self.all_agent_files[idx[0]]
            _, _, demo_file = self.all_demo_files[idx[1]]

        # if agent_file not in self.all_file_pairs:
        #     self._frame_distribution[agent_file] = np.zeros((1, 250))
        start = time.time()
        agent_traj = load_traj(agent_file)
        end = time.time()
        logger.debug(f"Loading time {end-start}")
        # start = time.time()
        # end = time.time()
        # print(f"Make demo {end-start}")
        # start = time.time()
        traj = self._make_traj(
            agent_traj[0],
            agent_traj[1],
            task_name,
            sub_task_id)
        # end = time.time()
        # print(f"Make traj {end-start}")
        return {'traj': traj, 'task_name': task_name, 'task_id': sub_task_id}

    def _make_traj(self, traj, command, task_name, sub_task_id):

        ret_dict = {}

        end = len(traj)
        start = torch.randint(low=1, high=max(
            1, end - self._obs_T + 1), size=(1,))

        if self._take_first_frame:
            first_frame = [torch.tensor(1)]
            chosen_t = first_frame + [j + start for j in range(self._obs_T)]
        else:
            chosen_t = [j + start for j in range(self._obs_T)]

        if self.non_sequential:
            chosen_t = torch.randperm(end)
            chosen_t = chosen_t[chosen_t != 0][:self._obs_T]

        images, images_cp, bb, obj_classes, action, states, points = create_sample(
            dataset_loader=self,
            traj=traj,
            chosen_t=chosen_t,
            task_name=task_name,
            command=command,
            load_action=True,
            load_state=self._load_state_spec,
            load_eef_point=self._load_eef_point)

        ret_dict['images'] = torch.stack(images)

        if self.aug_twice:
            ret_dict['images_cp'] = torch.stack(images_cp)

        ret_dict['gt_bb'] = torch.stack(bb)
        ret_dict['gt_classes'] = torch.stack(obj_classes)

        ret_dict['states'] = []
        ret_dict['states'] = np.array(states)

        ret_dict['actions'] = []
        ret_dict['actions'] = np.array(action)

        ret_dict['points'] = []
        ret_dict['points'] = np.array(points)

        if self.aux_pose:
            grip_close = np.array(
                [traj.get(i, False)['action'][-1] > 0 for i in range(1, len(traj))])
            grip_t = np.argmax(grip_close)
            drop_t = len(traj) - 1 - \
                np.argmax(np.logical_not(grip_close)[::-1])
            aux_pose = [traj.get(t, False)['obs']['ee_aa'][:3]
                        for t in (grip_t, drop_t)]
            ret_dict['aux_pose'] = np.concatenate(aux_pose).astype(np.float32)
        return ret_dict
