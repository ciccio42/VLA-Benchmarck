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


class MultiTaskPairedTargetObjDataset(Dataset):
    def __init__(
            self,
            tasks_spec,
            root_dir='mosaic_multitask_dataset',
            mode='train',
            split=[0.9, 0.1],
            aug_twice=True,
            width=180,
            height=100,
            demo_T=4,
            obs_T=1,
            data_augs=None,
            non_sequential=False,
            allow_val_skip=False,
            allow_train_skip=False,
            use_strong_augs=False,
            aux_pose=False,
            select_random_frames=True,
            compute_obj_distribution=False,
            agent_name='ur5e',
            demo_name='panda',
            load_action=False,
            ** params):

        self.task_crops = OrderedDict()
        # each idx i maps to a unique tuple of (task_name, sub_task_id, agent.pkl, demo.pkl)
        self.all_file_pairs = OrderedDict()
        count = 0
        self.task_to_idx = defaultdict(list)
        self.subtask_to_idx = OrderedDict()
        self.agent_files = dict()
        self.demo_files = dict()
        self.mode = mode
        self._demo_T = demo_T
        self._obs_T = obs_T

        self.select_random_frames = select_random_frames
        self.compute_obj_distribution = compute_obj_distribution
        self.object_distribution = OrderedDict()
        self.object_distribution_to_indx = OrderedDict()
        self.index_to_slot = OrderedDict()

        create_train_val_dict(self,
                              agent_name,
                              demo_name,
                              root_dir,
                              tasks_spec,
                              split,
                              allow_train_skip,
                              allow_val_skip)

        self.pairs_count = count
        self.task_count = len(tasks_spec)

        self.width, self.height = width, height
        self.aug_twice = aug_twice
        self.aux_pose = aux_pose

        self.non_sequential = non_sequential
        if non_sequential:
            print("Warning! The agent observations are not sampled in neighboring timesteps, make sure inverse dynamics loss is NOT used in training \n ")

        self.use_strong_augs = use_strong_augs
        self.data_augs = data_augs
        self.frame_aug = create_data_aug(self)

    def __len__(self):
        """NOTE: we should count total possible demo-agent pairs, not just single-file counts
        total pairs should sum over all possible sub-task pairs"""
        return self.pairs_count

    def __getitem__(self, idx):
        """since the data is organized by task, use a mapping here to convert
        an index to a proper sub-task index """
        if self.mode == 'train':
            pass
        (task_name, sub_task_id, demo_file,
         agent_file) = self.all_file_pairs[idx]
        demo_traj, agent_traj = load_traj(demo_file), load_traj(agent_file)
        demo_data = make_demo(self, demo_traj[0], task_name)
        traj = self._make_traj(agent_traj[0], task_name, idx)
        return {'demo_data': demo_data, 'traj': traj, 'task_name': task_name, 'task_id': sub_task_id}

    def _make_traj(self, traj, task_name, indx):
        # get the first frame from the trajectory
        ret_dict = {}
        images = []
        images_cp = []

        if self.non_sequential and self._obs_T > 1:
            # Select random frames
            end = len(traj)
            chosen_t = torch.randperm(end)
            chosen_t = chosen_t[chosen_t != 0][:self._obs_T]
            for j, t in enumerate(chosen_t):
                t = t.item()
                step_t = traj.get(t)
                image = copy.copy(
                    step_t['obs']['camera_front_image'][:, :, ::-1])
                processed = self.frame_aug(task_name, image)
                images.append(processed)
                if self.aug_twice:
                    images_cp.append(self.frame_aug(task_name, image, True))
        elif not self.non_sequential and self._obs_T > 1:
            end = len(traj)
            start = torch.randint(low=1, high=max(
                1, end - self._obs_T + 1), size=(1,))
            chosen_t = [j + start for j in range(self._obs_T)]
            images = []
            images_cp = []
            for j, t in enumerate(chosen_t):
                t = t.item()
                step_t = traj.get(t)
                image = copy.copy(
                    step_t['obs']['camera_front_image'][:, :, ::-1])
                processed = self.frame_aug(task_name, image)
                images.append(processed)
                if self.aug_twice:
                    images_cp.append(self.frame_aug(task_name, image, True))
        else:
            image = copy.copy(
                traj.get(1)['obs']['camera_front_image'][:, :, ::-1])
            images.append(self.frame_aug(task_name, image))
            images_cp.append(self.frame_aug(task_name, image, True))

        ret_dict['images'] = torch.stack(images)
        if self.aug_twice:
            ret_dict['images_cp'] = torch.stack(images_cp)
        # get target object position
        one_hot_encoding = np.zeros(4)
        one_hot_encoding[self.index_to_slot[indx]] = 1
        one_hot_encoding = torch.from_numpy(
            one_hot_encoding).repeat(self._obs_T, 1)
        ret_dict['target_position_one_hot'] = one_hot_encoding
        return ret_dict


class DIYBatchSampler(Sampler):
    """
    Customize any possible combination of both task families and sub-tasks in a batch of data.
    """

    def __init__(
        self,
        task_to_idx,
        subtask_to_idx,
        object_distribution_to_indx,
        sampler_spec=dict(),
        tasks_spec=dict(),
        n_step=0,
    ):
        """
        Args:
        - batch_size:
            total number of samples draw at each yield step
        - task_to_idx: {
            task_name: [all_idxs_for this task]}
        - sub_task_to_idx: {
            task_name: {
                {sub_task_id: [all_idxs_for this sub-task]}}
           all indics in both these dict()'s should sum to the total dataset size,
        - tasks_spec:
            should additionally contain batch-constructon guide:
            explicitly specify how to contruct the batch, use this spec we should be
            able to construct a mapping from each batch index to a fixed pair
            of [task_name, subtask_id] to sample from,
            but if set shuffle=true, the sampled batch would lose this ordering,
            e.g. give a _list_: ['${place}', '${nut_hard}']
            batch spec is extracted from:
                {'place':
                        {'task_ids':     [0,1,2],
                        'n_per_task':    [5, 10, 5]}
                'nut_hard':
                        {'task_ids':     [4],
                        'n_per_task':    [6]}
                'stack':
                        {...}
                }
                will yield a batch of 36 points, where first 5 comes from pickplace subtask#0, last 6 comes from nut-assembly task#4
        - shuffle:
            if true, we lose control over how each batch is distributed to gpus
        """
        batch_size = sampler_spec.get('batch_size', 30)
        drop_last = sampler_spec.get('drop_last', False)
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        self.shuffle = sampler_spec.get('shuffle', False)

        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        self.task_samplers = OrderedDict()
        self.task_iterators = OrderedDict()
        self.task_info = OrderedDict()
        self.balancing_policy = sampler_spec.get('balancing_policy', 0)
        self.object_distribution_to_indx = object_distribution_to_indx
        self.num_step = n_step
        for spec in tasks_spec:
            task_name = spec.name
            idxs = task_to_idx.get(task_name)
            self.task_samplers[task_name] = OrderedDict(
                {'all_sub_tasks': SubsetRandomSampler(idxs)})  # uniformly draw from union of all sub-tasks
            self.task_iterators[task_name] = OrderedDict(
                {'all_sub_tasks': iter(SubsetRandomSampler(idxs))})
            assert task_name in subtask_to_idx.keys(), \
                'Mismatch between {} task idxs and subtasks!'.format(task_name)
            num_loaded_sub_tasks = len(subtask_to_idx[task_name].keys())
            first_id = list(subtask_to_idx[task_name].keys())[0]

            sub_task_size = len(subtask_to_idx[task_name].get(first_id))
            print("Task {} loaded {} subtasks, starting from {}, should all have sizes {}".format(
                task_name, num_loaded_sub_tasks, first_id, sub_task_size))
            for sub_task, sub_idxs in subtask_to_idx[task_name].items():

                # the balancing has been requested
                if self.balancing_policy == 1 and self.object_distribution_to_indx != None:
                    self.task_samplers[task_name][sub_task] = [SubsetRandomSampler(
                        sample_list) for sample_list in object_distribution_to_indx[task_name][sub_task]]
                    self.task_iterators[task_name][sub_task] = [iter(SubsetRandomSampler(
                        sample_list)) for sample_list in object_distribution_to_indx[task_name][sub_task]]
                    for i, sample_list in enumerate(object_distribution_to_indx[task_name][sub_task]):
                        if len(sample_list) == 0:
                            print(
                                f"Task {task_name} - Sub-task {sub_task} - Position {i}")

                else:
                    self.task_samplers[task_name][sub_task] = SubsetRandomSampler(
                        sub_idxs)
                    assert len(sub_idxs) == sub_task_size, \
                        'Got uneven data sizes for sub-{} under the task {}!'.format(
                            sub_task, task_name)
                    self.task_iterators[task_name][sub_task] = iter(
                        SubsetRandomSampler(sub_idxs))
                    # print('subtask indexs:', sub_task, max(sub_idxs))
            curr_task_info = {
                'size':         len(idxs),
                'n_tasks':      len(subtask_to_idx[task_name].keys()),
                'sub_id_to_name': {i: name for i, name in enumerate(subtask_to_idx[task_name].keys())},
                'traj_per_subtask': sub_task_size,
                'sampler_len': -1  # to be decided below
            }
            self.task_info[task_name] = curr_task_info

        n_tasks = len(self.task_samplers.keys())
        n_total = sum([info['size'] for info in self.task_info.values()])

        self.idx_map = OrderedDict()
        idx = 0
        for spec in tasks_spec:
            name = spec.name
            _ids = spec.get('task_ids', None)
            n = spec.get('n_per_task', None)
            assert (
                _ids and n), 'Must specify which subtask ids to use and how many is contained in each batch'
            info = self.task_info[name]
            subtask_names = info.get('sub_id_to_name')
            for _id in _ids:
                subtask = subtask_names[_id]
                for _ in range(n):
                    self.idx_map[idx] = (name, subtask)
                    idx += 1
                sub_length = int(info['traj_per_subtask'] / n)
                self.task_info[name]['sampler_len'] = max(
                    sub_length, self.task_info[name]['sampler_len'])
        # print("Index map:", self.idx_map)

        self.max_len = max([info['sampler_len']
                           for info in self.task_info.values()])
        print('Max length for sampler iterator:', self.max_len)
        self.n_tasks = n_tasks

        assert idx == batch_size, "The constructed batch size {} doesn't match desired {}".format(
            idx, batch_size)
        self.batch_size = idx
        self.drop_last = drop_last

        print("Shuffling to break the task ordering in each batch? ", self.shuffle)

    def __iter__(self):
        """Given task families A,B,C, each has sub-tasks A00, A01,...
        Fix a total self.batch_size, sample different numbers of datapoints from
        each task"""
        batch = []
        for i in range(self.max_len):
            for idx in range(self.batch_size):
                (name, sub_task) = self.idx_map[idx]
                if self.balancing_policy == 1 and self.object_distribution_to_indx != None:
                    slot_indx = idx % len(self.task_samplers[name][sub_task])
                    # take one sample for the current task, sub_task, and slot
                    sampler = self.task_samplers[name][sub_task][slot_indx]
                    iterator = self.task_iterators[name][sub_task][slot_indx]
                    try:
                        batch.append(next(iterator))
                    except StopIteration:  # print('early sstop:', i, name)
                        # re-start the smaller-sized tasks
                        iterator = iter(sampler)
                        batch.append(next(iterator))
                        self.task_iterators[name][sub_task][slot_indx] = iterator
                else:
                    # print(name, sub_task)
                    sampler = self.task_samplers[name][sub_task]
                    iterator = self.task_iterators[name][sub_task]
                    try:
                        batch.append(next(iterator))
                    except StopIteration:  # print('early sstop:', i, name)
                        # re-start the smaller-sized tasks
                        iterator = iter(sampler)
                        batch.append(next(iterator))
                        self.task_iterators[name][sub_task] = iterator

            if len(batch) == self.batch_size:
                if self.shuffle:
                    random.shuffle(batch)
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            if self.shuffle:
                random.shuffle(batch)
            yield batch

    def __len__(self):
        # Since different task may have different data sizes,
        # define total length of sampler as number of iterations to
        # exhaust the last task
        return self.max_len
