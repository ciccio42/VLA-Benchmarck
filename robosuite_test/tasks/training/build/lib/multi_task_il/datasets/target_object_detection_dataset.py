import random
import torch
from os.path import join, expanduser
from multi_task_il.datasets import load_traj, split_files
import cv2
from torch.utils.data import Dataset, Sampler, SubsetRandomSampler, RandomSampler, WeightedRandomSampler
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from torchvision.transforms import RandomAffine, ToTensor, Normalize, \
    RandomGrayscale, ColorJitter, RandomApply, RandomHorizontalFlip, GaussianBlur, RandomResizedCrop
from torchvision.transforms.functional import resized_crop

import pickle as pkl
from collections import defaultdict, OrderedDict
import glob
import numpy as np
import matplotlib.pyplot as plt
import copy
from copy import deepcopy
from functools import reduce
from operator import concat
from multi_task_il.utils import normalize_action

ENV_OBJECTS = {
    'pick_place': {
        'obj_names': ['greenbox', 'yellowbox', 'bluebox', 'redbox'],
        'ranges': [[0.195, 0.255], [0.045, 0.105], [-0.105, -0.045], [-0.255, -0.195]],
    },
    'nut_assembly': {
        'obj_names': ['nut0', 'nut1', 'nut2'],
        'ranges': [[0.10, 0.31], [-0.10, 0.10], [-0.31, -0.10]]
    }
}

JITTER_FACTORS = {'brightness': 0.4,
                  'contrast': 0.4, 'saturation': 0.4, 'hue': 0.1}


def collate_by_task(batch):
    """ Use this for validation: groups data by task names to compute per-task losses """
    per_task_data = defaultdict(list)
    for b in batch:
        per_task_data[b['task_name']].append(
            {k: v for k, v in b.items() if k != 'task_name' and k != 'task_id'}
        )

    for name, data in per_task_data.items():
        per_task_data[name] = default_collate(data)
    return per_task_data


class TargetObjectDataset(Dataset):
    def __init__(
            self,
            tasks_spec,
            root_dir='mosaic_multitask_dataset',
            mode='train',
            split=[0.9, 0.1],
            demo_T=4,
            obs_T=7,
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
            normalization_ranges=[],
            n_action_bin=256,
            ** params):
        """
        Args:
        -  root_dir:
            path to robosuite multi-task's data folder e.g. /home/mandi/robosuite/multi_task
        -  tasks_spec:
            a **List** specifying data location and other info for each task
            e.g. task_name = 'place'
                tasks_spec[0] = {
                    'name':             'place'
                    'date':             '0418'
                    'crop':             [30, 70, 85, 85], # this is pre-data-aug cropping
                    'traj_per_subtask': 100,
                    'n_tasks':          16,
                    'skip_id':          [-1] # a list of task ids to exclude!
                }
        (below are info shared across tasks:)
        - height， width
            crop params can be different but keep final image sizes the same
        - demo_T, obs_T:
            fixed demontration length and observation length
        - data_augs:
            specify how to crop/translate/jitter the data _after_ each image is cropped into the same sizes
            e.g. {
                'rand_trans': 0.1,      # proportionally shift the image by 0.1 times its height/width
                'jitter': 0.5,    # probability _factor_ to jitter each of the four jitter factors
                'grayscale': 0.2,       # got recommended 0.2 or 0.1
                }
        - state_spec:
            which state vectors to extract
                e.g. ('ee_aa', 'ee_vel', 'joint_pos', 'joint_vel', 'gripper_qpos', 'object_detected')
        -  action_spec
                action keys to get
        -  allow_train_skip, allow_val_skip:
                whether we entirely skip loading some of the subtasks to the dataset
        -   non_sequential：
                whether to take strides when we sample， note if we do this then inverse dynamics loss is invalid
        """
        self.task_crops = OrderedDict()
        # each idx i maps to a unique tuple of (task_name, sub_task_id, agent.pkl, demo.pkl)
        self.all_file_pairs = OrderedDict()
        count = 0
        self.task_to_idx = defaultdict(list)
        self.subtask_to_idx = OrderedDict()
        self.agent_files = dict()
        self.demo_files = dict()
        self.mode = mode

        self.select_random_frames = select_random_frames
        self.balance_target_obj_pos = balance_target_obj_pos
        self.compute_obj_distribution = compute_obj_distribution
        self.object_distribution = OrderedDict()
        self.object_distribution_to_indx = OrderedDict()

        self._take_first_frame = take_first_frame

        self._selected_target_frame_distribution_task_object_target_position = OrderedDict()

        self._compute_frame_distribution = False
        self._normalize_action = normalize_action
        self._normalization_ranges = np.array(normalization_ranges)
        self._n_action_bin = n_action_bin

        # Frame distribution for each trajectory
        self._frame_distribution = OrderedDict()

        for spec in tasks_spec:
            name, date = spec.get('name', None), spec.get('date', None)
            assert name, 'need to specify the task name for data generated, for easier tracking'
            self.agent_files[name] = dict()
            self.demo_files[name] = dict()

            self.object_distribution[name] = OrderedDict()
            self.object_distribution_to_indx[name] = OrderedDict()

            if mode == 'train':
                print(
                    "Loading task [{:<9}] saved on date {}".format(name, date))
            if date is None:
                agent_dir = join(
                    root_dir, name, '{}_{}'.format(agent_name, name))
                demo_dir = join(
                    root_dir, name, '{}_{}'.format(demo_name, name))
            else:
                agent_dir = join(
                    root_dir, name, '{}_{}_{}'.format(date, agent_name, name))
                demo_dir = join(
                    root_dir, name, '{}_{}_{}'.format(date, demo_name, name))
            self.subtask_to_idx[name] = defaultdict(list)
            for _id in range(spec.get('n_tasks')):
                if _id in spec.get('skip_ids', []):
                    if (allow_train_skip and mode == 'train') or (allow_val_skip and mode == 'val'):
                        print(
                            'Warning! Excluding subtask id {} from loaded **{}** dataset for task {}'.format(_id, mode, name))
                        continue
                task_id = 'task_{:02d}'.format(_id)
                task_dir = expanduser(join(agent_dir,  task_id, '*.pkl'))
                agent_files = sorted(glob.glob(task_dir))
                assert len(agent_files) != 0, "Can't find dataset for task {}, subtask {} in dir {}".format(
                    name, _id, task_dir)
                subtask_size = spec.get('traj_per_subtask', 100)
                assert len(
                    agent_files) >= subtask_size, "Doesn't have enough data "+str(len(agent_files))
                agent_files = agent_files[:subtask_size]

                # prev. version does split randomly, here we strictly split each subtask in the same split ratio:
                idxs = split_files(len(agent_files), split, mode)
                agent_files = [agent_files[i] for i in idxs]

                task_dir = expanduser(join(demo_dir, task_id, '*.pkl'))

                demo_files = sorted(glob.glob(task_dir))
                subtask_size = spec.get('demo_per_subtask', 100)
                assert len(
                    demo_files) >= subtask_size, "Doesn't have enough data "+str(len(demo_files))
                demo_files = demo_files[:subtask_size]
                idxs = split_files(len(demo_files), split, mode)
                demo_files = [demo_files[i] for i in idxs]
                # assert len(agent_files) == len(demo_files), \
                #     'data for task {}, subtask #{} is not matched'.format(name, task_id)

                self.agent_files[name][_id] = deepcopy(agent_files)
                self.demo_files[name][_id] = deepcopy(demo_files)

                self.object_distribution[name][task_id] = OrderedDict()
                self.object_distribution_to_indx[name][task_id] = [
                    [] for i in range(len(ENV_OBJECTS[name]['ranges']))]
                if self.compute_obj_distribution and self.mode == 'train':
                    # for each subtask, create a dict with the object name
                    # assign the slot at each file
                    for agent in agent_files:
                        # compute object distribution if requested
                        if self.compute_obj_distribution:
                            # load pickle file
                            with open(agent, "rb") as f:
                                agent_file_data = pkl.load(f)
                            # take trj
                            trj = agent_file_data['traj']
                            # take target object id
                            target_obj_id = trj[1]['obs']['target-object']
                            for id, obj_name in enumerate(ENV_OBJECTS[name]['obj_names']):
                                if id == target_obj_id:
                                    if obj_name not in self.object_distribution[name][task_id]:
                                        self.object_distribution[name][task_id][obj_name] = OrderedDict(
                                        )
                                    # get object position
                                    if name == 'nut_assembly':
                                        if id == 0:
                                            pos = trj[1]['obs']['round-nut_pos']
                                        else:
                                            pos = trj[1]['obs'][f'round-nut-{id+1}_pos']
                                    else:
                                        pos = trj[1]['obs'][f'{obj_name}_pos']
                                    for i, pos_range in enumerate(ENV_OBJECTS[name]["ranges"]):
                                        if pos[1] >= pos_range[0] and pos[1] <= pos_range[1]:
                                            self.object_distribution[name][task_id][obj_name][agent] = i
                                            break
                                    break

                for demo in demo_files:
                    for agent in agent_files:
                        self.all_file_pairs[count] = (name, _id, demo, agent)
                        self.task_to_idx[name].append(count)
                        self.subtask_to_idx[name][task_id].append(count)
                        if self.compute_obj_distribution and self.mode == 'train':
                            # take objs for the current task_id
                            for obj in self.object_distribution[name][task_id].keys():
                                # take the slot for the given agent file
                                if agent in self.object_distribution[name][task_id][obj]:
                                    slot_indx = self.object_distribution[name][task_id][obj][agent]
                                    # assign the slot for the given agent file
                                    self.object_distribution_to_indx[name][task_id][slot_indx].append(
                                        count)
                        count += 1

            self.task_crops[name] = spec.get('crop', [0, 0, 0, 0])

        print('Done loading Task {}, agent/demo trajctores pairs reach a count of: {}'.format(name, count))
        self.pairs_count = count
        self.task_count = len(tasks_spec)

        self._demo_T, self._obs_T = demo_T, obs_T
        self.width, self.height = width, height
        self.aug_twice = aug_twice
        self.aux_pose = aux_pose

        self._state_action_spec = (state_spec, action_spec)
        self.non_sequential = non_sequential
        if non_sequential:
            print("Warning! The agent observations are not sampled in neighboring timesteps, make sure inverse dynamics loss is NOT used in training \n ")

        assert data_augs, 'Must give some basic data-aug parameters'
        if mode == 'train':
            print('Data aug parameters:', data_augs)
        # self.randAffine = RandomAffine(degrees=0, translate=(data_augs.get('rand_trans', 0.1), data_augs.get('rand_trans', 0.1)))
        self.toTensor = ToTensor()
        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        jitters = {k: v * data_augs.get('weak_jitter', 0)
                   for k, v in JITTER_FACTORS.items()}
        weak_jitter = ColorJitter(**jitters)

        weak_scale = data_augs.get('weak_crop_scale', (0.8, 1.0))
        weak_ratio = data_augs.get('weak_crop_ratio', (1.6, 1.8))
        randcrop = RandomResizedCrop(
            size=(height, width), scale=weak_scale, ratio=weak_ratio)
        if data_augs.use_affine:
            randcrop = RandomAffine(degrees=0, translate=(data_augs.get(
                'rand_trans', 0.1), data_augs.get('rand_trans', 0.1)))
        self.transforms = transforms.Compose([  # normalize at the end
            RandomApply([weak_jitter], p=0.1),
            RandomApply(
                [GaussianBlur(kernel_size=5, sigma=data_augs.get('blur', (0.1, 2.0)))], p=0.1),
            randcrop,
            # self.normalize
        ])

        self.use_strong_augs = use_strong_augs
        print("Using strong augmentations?", use_strong_augs)
        jitters = {k: v * data_augs.get('strong_jitter', 0)
                   for k, v in JITTER_FACTORS.items()}
        strong_jitter = ColorJitter(**jitters)
        self.grayscale = RandomGrayscale(data_augs.get("grayscale", 0))
        strong_scale = data_augs.get('strong_crop_scale', (0.2, 0.76))
        strong_ratio = data_augs.get('strong_crop_ratio', (1.2, 1.8))
        self.strong_augs = transforms.Compose([
            RandomApply([strong_jitter], p=0.05),
            self.grayscale,
            RandomHorizontalFlip(p=data_augs.get('flip', 0)),
            RandomApply(
                [GaussianBlur(kernel_size=5, sigma=data_augs.get('blur', (0.1, 2.0)))], p=0.01),
            RandomResizedCrop(
                size=(height, width), scale=strong_scale, ratio=strong_ratio),
            # self.normalize,
        ])

        def frame_aug(task_name, obs, second=False):
            """applies to every timestep's RGB obs['camera_front_image']"""
            crop_params = self.task_crops.get(task_name, [0, 0, 0, 0])
            top, left = crop_params[0], crop_params[2]
            img_height, img_width = obs.shape[0], obs.shape[1]
            box_h, box_w = img_height - top - \
                crop_params[1], img_width - left - crop_params[3]

            obs = self.toTensor(obs)
            # only this resize+crop is task-specific
            obs = resized_crop(obs, top=top, left=left, height=box_h,
                               width=box_w, size=(self.height, self.width))
            # cv2.imwrite("resized.png", np.moveaxis(
            #     obs.numpy()*255, 0, -1))
            if self.use_strong_augs and second:
                augmented = self.strong_augs(obs)
            else:
                augmented = self.transforms(obs)
            assert augmented.shape == obs.shape

            # if self.mode == 'val':
            #     cv2.imwrite(f"augment_val.png", np.moveaxis(
            #         augmented.numpy()*255, 0, -1))
            # if self.mode == 'train':
            #     cv2.imwrite(f"augment_train.png", np.moveaxis(
            #         augmented.numpy()*255, 0, -1))
            return augmented
        self.frame_aug = frame_aug

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

        if agent_file not in self.all_file_pairs:
            self._frame_distribution[agent_file] = np.zeros((1, 250))

        demo_traj, agent_traj = load_traj(demo_file), load_traj(agent_file)
        demo_data = self._make_demo(demo_traj[0], task_name)
        traj = self._make_traj(agent_traj[0], task_name, agent_file)
        return {'demo_data': demo_data, 'traj': traj, 'task_name': task_name, 'task_id': sub_task_id}

    def _make_demo(self, traj, task_name):
        """
        Do a near-uniform sampling of the demonstration trajectory
        """
        if self.select_random_frames:
            def clip(x): return int(max(0, min(x, len(traj) - 1)))
            per_bracket = max(len(traj) / self._demo_T, 1)
            frames = []
            cp_frames = []
            for i in range(self._demo_T):
                # fix to using uniform + 'sample_side' now
                if i == self._demo_T - 1:
                    n = len(traj) - 1
                elif i == 0:
                    n = 1
                else:
                    n = clip(np.random.randint(
                        int(i * per_bracket), int((i + 1) * per_bracket)))
                # frames.append(_make_frame(n))
                # convert from BGR to RGB and scale to 0-1 range
                obs = copy.copy(
                    traj.get(n)['obs']['camera_front_image'][:, :, ::-1])
                processed = self.frame_aug(task_name, obs)
                frames.append(processed)
                if self.aug_twice:
                    cp_frames.append(self.frame_aug(task_name, obs, True))
        else:
            frames = []
            cp_frames = []
            for i in range(self._demo_T):
                # get first frame
                if i == 0:
                    n = 0
                # get the last frame
                elif i == self._demo_T - 1:
                    n = len(traj) - 1
                elif i == 1:
                    obj_in_hand = 0
                    # get the first frame with obj_in_hand and the gripper is closed
                    for t in range(1, len(traj)):
                        state = traj.get(t)['info']['status']
                        trj_t = traj.get(t)
                        gripper_act = trj_t['action'][-1]
                        if state == 'obj_in_hand' and gripper_act == 1:
                            obj_in_hand = t
                            n = t
                            break
                elif i == 2:
                    # get the middle moving frame
                    start_moving = 0
                    end_moving = 0
                    for t in range(obj_in_hand, len(traj)):
                        state = traj.get(t)['info']['status']
                        if state == 'moving' and start_moving == 0:
                            start_moving = t
                        elif state != 'moving' and start_moving != 0 and end_moving == 0:
                            end_moving = t
                            break
                    n = start_moving + int((end_moving-start_moving)/2)

                # convert from BGR to RGB and scale to 0-1 range
                obs = copy.copy(
                    traj.get(n)['obs']['camera_front_image'][:, :, ::-1])

                processed = self.frame_aug(task_name, obs)
                frames.append(processed)
                if self.aug_twice:
                    cp_frames.append(self.frame_aug(task_name, obs, True))

        ret_dict = dict()
        ret_dict['demo'] = torch.stack(frames)
        ret_dict['demo_cp'] = torch.stack(cp_frames)
        return ret_dict

    def _make_traj(self, traj, task_name, agent_file):
        crop_params = self.task_crops.get(task_name, [0, 0, 0, 0])

        def _adjust_points(points, frame_dims):
            h = np.clip(points[0] - crop_params[0], 0,
                        frame_dims[0] - crop_params[1])
            w = np.clip(points[1] - crop_params[2], 0,
                        frame_dims[1] - crop_params[3])
            h = float(
                h) / (frame_dims[0] - crop_params[0] - crop_params[1]) * self.height
            w = float(
                w) / (frame_dims[1] - crop_params[2] - crop_params[3]) * self.width
            return tuple([int(min(x, d - 1)) for x, d in zip([h, w], (self.height, self.width))])

        def _get_tensor(k, step_t):
            if k == 'action':
                return step_t['action']
            elif k == 'grip_action':
                return [step_t['action'][-1]]
            o = step_t['obs']
            if k == 'ee_aa' and 'ee_aa' not in o:
                ee, axis_angle = o['ee_pos'][:3], o['axis_angle']
                if axis_angle[0] < 0:
                    axis_angle[0] += 2
                o = np.concatenate((ee, axis_angle)).astype(np.float32)
            else:
                o = o[k]
            return o

        state_keys, action_keys = self._state_action_spec
        ret_dict = {'states': [], 'actions': []}
        has_eef_point = 'eef_point' in traj.get(0, False)['obs']
        if has_eef_point:
            ret_dict['points'] = []
        end = len(traj)
        start = torch.randint(low=1, high=max(
            1, end - self._obs_T + 1), size=(1,))

        if self._take_first_frame:
            first_frame = [torch.tensor(1)]
            chosen_t = first_frame

        images = []
        images_cp = []

        for j, t in enumerate(chosen_t):

            # self._frame_distribution[agent_file][t] = self._frame_distribution[agent_file][t]

            t = t.item()

            step_t = traj.get(t)
            image = copy.copy(
                step_t['obs']['camera_front_image'][:, :, ::-1])
            processed = self.frame_aug(task_name, image)
            images.append(processed)
            if self.aug_twice:
                images_cp.append(self.frame_aug(task_name, image, True))
            if has_eef_point:
                ret_dict['points'].append(np.array(
                    _adjust_points(step_t['obs']['eef_point'], image.shape[:2]))[None])

            state = []
            for k in state_keys:
                state.append(_get_tensor(k, step_t))
            ret_dict['states'].append(
                np.concatenate(state).astype(np.float32)[None])

            if (j >= 1 and not self._take_first_frame) or (self._take_first_frame and j >= 2):
                action = []
                for k in action_keys:
                    action.append(_get_tensor(k, step_t))

                if self._normalize_action:
                    action = normalize_action(
                        action=action[0],
                        n_action_bin=self._n_action_bin,
                        action_ranges=self._normalization_ranges
                    )[None]

                ret_dict['actions'].append(
                    np.concatenate(action).astype(np.float32)[None])

        for k, v in ret_dict.items():
            ret_dict[k] = np.concatenate(v, 0).astype(np.float32)

        ret_dict['images'] = torch.stack(images)
        if self.aug_twice:
            ret_dict['images_cp'] = torch.stack(images_cp)

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
        self.batch_size = batch_size
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
