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
from multi_task_il.datasets.data_aug import DataAugmentation

class MultiTaskPairedDataset(Dataset):
    def __init__(
            self,
            tasks_spec,
            root_dir='mosaic_multitask_dataset',
            mode='train',
            split=[0.9, 0.1],
            demo_T=4,
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
            demo_name='panda',
            normalize_action=True,
            pick_next=False,
            normalization_ranges=[],
            n_action_bin=256,
            perform_augs=True,
            perform_scale_resize=True,
            change_command_epoch=True,
            load_eef_point=False,
            split_pick_place=False,
            mix_sim_real=False,
            convert_action=False,
            dagger=False,
            ** params):

        self.task_crops = OrderedDict()
        self.demo_crop = OrderedDict()
        self.agent_crop = OrderedDict()
        self.agent_sim_crop = OrderedDict()
        # each idx i maps to a unique tuple of (task_name, sub_task_id, agent.pkl, demo.pkl)
        self.all_file_pairs = OrderedDict()
        self.all_agent_files = OrderedDict()
        self.all_demo_files = OrderedDict()
        self.task_to_idx = defaultdict(list)
        self.subtask_to_idx = OrderedDict()
        self.demo_task_to_idx = defaultdict(list)
        self.demo_subtask_to_idx = OrderedDict()
        self.agent_files = dict()
        self.demo_files = dict()
        self.agent_name = agent_name
        self.mode = mode
        self.pick_next = pick_next
        self.split_pick_place = split_pick_place

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
        self._mix_sim_real = mix_sim_real
        self._convert_action = convert_action
        self._dagger = dagger 

        # Frame distribution for each trajectory
        self._frame_distribution = OrderedDict()
        self._mix_demo_agent = False
        count, pairs_cnt = create_train_val_dict(self,
                                      agent_name,
                                      demo_name,
                                      root_dir,
                                      tasks_spec,
                                      split,
                                      allow_train_skip,
                                      allow_val_skip)

        self.pairs_cnt = pairs_cnt
        self.step_cnt = count

        self.task_count = len(tasks_spec)

        self._demo_T, self._obs_T, self._bbs_T,  self._action_T = demo_T, obs_T, bbs_T, action_T
        self.width, self.height = width, height
        self.aug_twice = aug_twice
        self.aux_pose = aux_pose

        self._state_action_spec = (state_spec, action_spec)
        self.non_sequential = non_sequential
        if non_sequential:
            print("Warning! The agent observations are not sampled in neighboring timesteps, make sure inverse dynamics loss is NOT used in training \n ")

        self.use_strong_augs = use_strong_augs
        self.data_augs = data_augs
        self.frame_aug = DataAugmentation(data_augs=data_augs,
                                          mode=mode,
                                          height=height,
                                          width=width,
                                          use_strong_augs=use_strong_augs,
                                          task_crops=self.task_crops,
                                          agent_sim_crop=self.agent_sim_crop,
                                          demo_crop=self.demo_crop,
                                          agent_crop=self.agent_crop,)

    def __len__(self):
        """NOTE: we should count total possible demo-agent pairs, not just single-file counts
        total pairs should sum over all possible sub-task pairs"""
        return self.step_cnt

    def __getitem__(self, idx):
        """since the data is organized by task, use a mapping here to convert
        an index to a proper sub-task index """
        if self.mode == 'train':
            pass
        demo_indx = idx[0]
        sample_idx = idx[1]
        frame_idx = idx[2]
        #(task_name, sub_task_id, demo_file, agent_file, traj_len) = self.all_file_pairs[sample_idx]
        (_, _, demo_file) = self.all_demo_files[demo_indx]
        (task_name, sub_task_id, agent_file, trj_len) = self.all_agent_files[sample_idx]
        
        human_demo = False
        if ("human" in demo_file):
            human_demo = True
        
        # if agent_file not in self.all_file_pairs:
        #     self._frame_distribution[agent_file] = np.zeros((1, 250))
        start = time.time()
        demo_traj, agent_traj = load_traj(demo_file), load_traj(agent_file)
        
        sim_crop = False
        if "real_new" not in agent_file:
            sim_crop = True
        
        end = time.time()
        logger.debug(f"Loading time {end-start}")
        # start = time.time()
        demo_data = make_demo(self, demo_traj[0], task_name, human_demo)
        # end = time.time()
        # print(f"Make demo {end-start}")
        # start = time.time()
        traj = self._make_traj(
            frame_idx,
            agent_traj[0],
            agent_traj[1],
            task_name,
            sub_task_id,
            sim_crop,
            self._convert_action,
            human_demo)
        # end = time.time()
        # print(f"Make traj {end-start}")
        # print(sub_task_id)
        return {'demo_data': demo_data, 'traj': traj, 'task_name': task_name, 'task_id': sub_task_id}

    def _make_traj(self, start_frame, traj, command, task_name, sub_task_id, sim_crop, convert_action, human_demo):

        ret_dict = {}

        end = len(traj)
        # if(start_frame == 0):
        #     start_frame = 1
        # start = start_frame if start_frame + self._obs_T < end else start_frame - (self._obs_T - (end - 1 - start_frame))
        
        # print("start frame with randint", start_frame)
        start = torch.randint(low=1, high=max(1, end - self._obs_T + 1), size=(1,))

        if self._take_first_frame:
            first_frame = [torch.tensor(1)]
            chosen_t = first_frame + [j + start for j in range(self._obs_T)]
        else:
            chosen_t = [j + start for j in range(self._obs_T)]

        if self.non_sequential:
            chosen_t = torch.randperm(end)
            chosen_t = chosen_t[chosen_t != 0][:self._obs_T]

        first_phase = None
        if self.split_pick_place:
            first_t = chosen_t[0] #.item()
            last_t = chosen_t[-1] #.item()
            if task_name == 'nut_assembly' or task_name == 'pick_place' or 'button' in task_name or 'stack_block' in task_name:
                first_step_gripper_state = traj.get(first_t)['action'][-1]
                first_phase = True if first_step_gripper_state == -1.0 or first_step_gripper_state == 0.0 else False
                last_step_gripper_state = traj.get(last_t)['action'][-1]

                # if first_step_gripper_state == 1.0 and last_step_gripper_state == -1.0:
                #     print("Last with placing")
                if (first_step_gripper_state != last_step_gripper_state) and not (first_step_gripper_state == 1.0 and (last_step_gripper_state == -1.0 or last_step_gripper_state == 0.0)):
                    # change in task phase
                    for indx, step in enumerate(range(first_t, last_t+1)):
                        action_t = traj.get(step)['action'][-1]
                        if first_step_gripper_state != action_t:
                            step_change = step
                            break
                    for indx, step in enumerate(range(step_change+1-self._obs_T, step_change+1)):
                        chosen_t[indx] = torch.tensor(step)

        images, images_cp, bb, obj_classes, action, states, points = create_sample(
            dataset_loader=self,
            traj=traj,
            chosen_t=chosen_t,
            task_name=task_name,
            command=command,
            load_action=True,
            load_state=self._load_state_spec,
            load_eef_point=self._load_eef_point,
            agent_task_id=sub_task_id,
            sim_crop=sim_crop,
            convert_action=convert_action,
            human_demo=human_demo)

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

        if self.split_pick_place:
            ret_dict['first_phase'] = torch.tensor(first_phase)

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
