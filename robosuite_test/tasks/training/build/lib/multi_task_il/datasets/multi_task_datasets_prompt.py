import random
import torch
import os
from os.path import join, expanduser
from multi_task_il.datasets import load_traj, split_files
import cv2
from torch.utils.data import Dataset, Sampler, SubsetRandomSampler, RandomSampler, WeightedRandomSampler
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from torchvision.transforms import RandomAffine, ToTensor, Normalize, \
    RandomGrayscale, ColorJitter, RandomApply, RandomHorizontalFlip, GaussianBlur, RandomResizedCrop
from torchvision.transforms.functional import resized_crop
from tokenizers import Tokenizer
from tokenizers import AddedToken

import pickle as pkl
from collections import defaultdict, OrderedDict
import glob
import numpy as np
import matplotlib.pyplot as plt
import copy
from copy import deepcopy
from functools import reduce
from operator import concat
from multi_task_il.utils import normalize_action, discretize_action
from einops import rearrange
from torchmetrics.classification import Accuracy

from multi_task_il.models.vima.utils import *


ENV_OBJECTS = {
    'pick_place': {
        'obj_names': ['greenbox', 'yellowbox', 'bluebox', 'redbox', 'bin'],
        'splitted_obj_names': ['green box', 'yellow box', 'blue box', 'red box'],
        'ranges': [[0.195, 0.255], [0.045, 0.105], [-0.105, -0.045], [-0.255, -0.195]],
    },
    'nut_assembly': {
        'obj_names': ['round-nut', 'round-nut-2', 'round-nut-3', "peg1", "peg2", "peg3"],
        'splitted_obj_names': ['grey nut', 'brown nut', 'blue nut'],
        'ranges': [[0.10, 0.31], [-0.10, 0.10], [-0.31, -0.10]]
    }
}

JITTER_FACTORS = {'brightness': 0.4,
                  'contrast': 0.4, 'saturation': 0.4, 'hue': 0.1}

os.environ["TOKENIZERS_PARALLELISM"] = "true"
_kwargs = {
    "single_word": True,
    "lstrip": False,
    "rstrip": False,
    "normalized": True,
}
PLACEHOLDER_TOKENS = [
    AddedToken("{pick_object}", **_kwargs),
]
PLACEHOLDERS = [token.content for token in PLACEHOLDER_TOKENS]
tokenizer = Tokenizer.from_pretrained("t5-base")
tokenizer.add_tokens(PLACEHOLDER_TOKENS)


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


class MultiTaskPairedDataset(Dataset):
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
            n_action_bin=[256],
            views=['front'],
            modality=['rgb'],
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
        self._n_action_bin = [100-1, 100-1, 100-1, 50-1, 50-1, 50-1, 2-1]
        self._views = views
        self._modality = modality
        # Frame distribution for each trajectory
        self._frame_distribution = OrderedDict()

        create_train_val_dict(self,
                              agent_name,
                              demo_name,
                              root_dir,
                              tasks_spec,
                              split,
                              allow_train_skip,
                              allow_val_skip)

        print('Done loading Task {}, agent/demo trajctores pairs reach a count of: {}'.format(name, count))
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
        (task_name, sub_task_id, agent_file) = self.all_file_pairs[idx]

        if agent_file not in self.all_file_pairs:
            self._frame_distribution[agent_file] = np.zeros((1, 250))

        agent_traj, command = load_traj(agent_file)
        sample = self._make_prompt(command, task_name, agent_traj)

        return {'sample': sample, 'task_name': task_name, 'task_id': sub_task_id}

    def _make_prompt(self, command: str, task_name: str, agent_traj: object, add_special_token: bool = True):
        """Generate sample for prompted model

        Args:
            command (str): _description_
            task_name (str): _description_
            agent_traj (object): _description_
        """
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

        ret_dict = {'states': [],
                    'actions': [],
                    'prompt': None}

        state_keys, action_keys = self._state_action_spec
        has_eef_point = 'eef_point' in agent_traj.get(0, False)['obs']

        if has_eef_point:
            end = len(agent_traj)
            start = torch.randint(low=1, high=max(
                1, end - self._obs_T + 1), size=(1,))

        # Select frames number
        if self._take_first_frame and self.select_random_frames:
            first_frame = [torch.tensor(1)]
            chosen_t = first_frame + \
                [j + start for j in range(self._obs_T)]

        elif (not self._take_first_frame) and self.select_random_frames:
            chosen_t = [j + start for j in range(self._obs_T)]

        elif not self.select_random_frames:
            chosen_t = []
            previous_status = None
            for t in range(len(agent_traj)):
                status = agent_traj.get(t)['info']['status']
                if t == 1:
                    # take the first valid frame
                    chosen_t.append(t)
                    previous_status = status
                elif status == 'moving' and previous_status == 'prepare_grasp':
                    # take the last obj_in_hand frame
                    chosen_t.append(t-1)
                    previous_status = "obj_in_hand"
                elif t == len(agent_traj)-1:
                    # the the last placing frame
                    chosen_t.append(t-1)
                    previous_status = status
            chosen_t = torch.from_numpy(np.array(chosen_t))

        if self.non_sequential:
            chosen_t = torch.randperm(end)
            chosen_t = chosen_t[chosen_t != 0][:self._obs_T]

        # Get desired frames
        for j, t in enumerate(chosen_t):

            t = t.item()
            step_t = agent_traj.get(t)

            state = []
            for k in state_keys:
                state.append(_get_tensor(k, step_t))
            ret_dict['states'].append(
                np.concatenate(state).astype(np.float32)[None])

            if j >= 1:
                action = []
                for k in action_keys:
                    action.append(_get_tensor(k, step_t))

                if self._normalize_action:
                    action = discretize_action(
                        action=action[0],
                        n_action_bin=self._n_action_bin,
                        action_ranges=self._normalization_ranges
                    )[None]

                ret_dict['actions'].append(
                    np.concatenate(action)[None])

        for k, v in ret_dict.items():
            if v is not None:
                ret_dict[k] = np.concatenate(v, 0).astype(np.float32)

        # [TODO Add special tokens]
        if add_special_token:
            if task_name == 'pick_place':
                # add special token
                color = command.split(" ")[2]
                object = command.split(" ")[3]
                command = command.replace(
                    f"{color} {object}", "{pick_object}")
                ret_dict['prompt'] = command

                prompt_assets = self._create_prompt_assets(
                    obs=agent_traj.get(1)['obs'],
                    task_name=task_name,
                    views=self._views,
                    modalities=self._modality
                )
                prompt_token_type, word_batch, image_batch = self._prepare_prompt(
                    obs=agent_traj.get(1)['obs'],
                    task_name=task_name,
                    prompt=command,
                    prompt_assets=prompt_assets,
                    views=self._views
                )
                ret_dict['prompt_token_type'] = prompt_token_type
                ret_dict['word_batch'] = word_batch
                ret_dict['image_batch'] = image_batch

                ret_dict['obs'] = self._prepare_obs(agent_traj=agent_traj,
                                                    chosen_t=chosen_t,
                                                    views=self._views,
                                                    task_name=task_name)
            elif task_name == 'nut_assembly':
                # print(command)
                # add special token
                color = command.split(" ")[2]
                object = command.split(" ")[3]
                command = command.replace(
                    f"{color} {object}", "{pick_object}")
                ret_dict['prompt'] = command

                prompt_assets = self._create_prompt_assets(
                    obs=agent_traj.get(1)['obs'],
                    task_name=task_name,
                    views=self._views,
                    modalities=self._modality
                )
                prompt_token_type, word_batch, image_batch = self._prepare_prompt(
                    obs=agent_traj.get(1)['obs'],
                    task_name=task_name,
                    prompt=command,
                    prompt_assets=prompt_assets,
                    views=self._views
                )
                ret_dict['prompt_token_type'] = prompt_token_type
                ret_dict['word_batch'] = word_batch
                ret_dict['image_batch'] = image_batch

                ret_dict['obs'] = self._prepare_obs(agent_traj=agent_traj,
                                                    chosen_t=chosen_t,
                                                    views=self._views,
                                                    task_name=task_name)

        else:
            ret_dict['prompt'] = command

        return ret_dict

    def _prepare_prompt(self, obs: object, task_name: str, prompt: str, prompt_assets: dict, views: list):
        views = sorted(views)
        encoding = tokenizer.encode(prompt, add_special_tokens=True)
        prompt_ids, prompt_tokens = encoding.ids, encoding.tokens
        assert set(prompt_assets.keys()) == set(
            [token[1:-1] for token in prompt_tokens if token in PLACEHOLDERS]
        )
        filled_prompt = []
        for id, token in zip(prompt_ids, prompt_tokens):
            if token not in PLACEHOLDERS:
                assert "{" not in token and "}" not in token
                filled_prompt.append(id)
            else:
                assert token.startswith("{") and token.endswith("}")
                asset_name = token[1:-1]
                assert asset_name in prompt_assets, f"missing prompt asset {asset_name}"
                asset = prompt_assets[asset_name]
                obj_info = asset["segm"]["obj_info"]
                placeholder_type = asset["placeholder_type"]
                if placeholder_type == "object":
                    objects = [obj_info["obj_id"]]
                elif placeholder_type == "scene":
                    objects = [each_info["obj_id"] for each_info in obj_info]
                obj_repr = {
                    "cropped_img": {view: [] for view in views},
                    "bbox": {view: [] for view in views},
                }

                for view in views:
                    modality_view = asset['rgb'][view]
                    bboxes = []
                    cropped_imgs = []
                    # for each object, crop the image around the target object
                    for i, obj_id in enumerate(objects):
                        # Create bounding box for the target object
                        if task_name == 'pick_place' or task_name == 'nut_assembly':
                            if i == 0:
                                # In pick-place the first object is the target object
                                target_obj_id = obs['target-object']
                                target_obj_name = ENV_OBJECTS[task_name]['obj_names'][target_obj_id]
                                target_obj_bb = obs['obj_bb']['camera_front'][target_obj_name]
                                upper_left_corner = target_obj_bb['upper_left_corner']
                                bottom_right_corner = target_obj_bb['bottom_right_corner']
                                object_center = target_obj_bb['center']
                                # get prompt observation
                                rgb_this_view = asset['rgb'][view]
                                prompt_img = cv2.rectangle(
                                    np.array(rgb_this_view), upper_left_corner, bottom_right_corner, (255, 0, 0), 1)
                                # cv2.imwrite("rgb_this_view.png",
                                #             np.array(prompt_img))

                                # bounding box center, height and width
                                x_center, y_center = object_center[0], object_center[1]
                                h, w = upper_left_corner[1] - \
                                    bottom_right_corner[1], upper_left_corner[0] - \
                                    bottom_right_corner[0]
                                bboxes.append(
                                    [int(x_center), int(y_center), int(h), int(w)])
                                # crop image
                                cropped_img = np.array(rgb_this_view[
                                    bottom_right_corner[1]:upper_left_corner[1] + 1, bottom_right_corner[0]:upper_left_corner[0] + 1, :])
                                # cv2.imwrite("cropped_img.png",
                                #             np.array(cropped_img))
                                # pad if dimensions are different
                                if cropped_img.shape[0] != cropped_img.shape[1]:
                                    diff = abs(
                                        cropped_img.shape[0] - cropped_img.shape[1])
                                    pad_before, pad_after = int(
                                        diff / 2), diff - int(diff / 2)
                                    if cropped_img.shape[0] > cropped_img.shape[1]:
                                        pad_width = (
                                            (0, 0), (pad_before, pad_after), (0, 0))
                                    else:
                                        pad_width = (
                                            (pad_before, pad_after), (0, 0), (0, 0))
                                    cropped_img = np.pad(
                                        cropped_img,
                                        pad_width,
                                        mode="constant",
                                        constant_values=0,
                                    )
                                    assert cropped_img.shape[0] == cropped_img.shape[1], "INTERNAL"
                                cropped_img = np.asarray(cropped_img)
                                # cv2.imwrite("cropped_img.png", cropped_img)
                                cropped_img = cv2.resize(
                                    cropped_img,
                                    (32, 32),
                                    interpolation=cv2.INTER_AREA,
                                )
                                cropped_img = rearrange(
                                    cropped_img, "h w c -> c h w")
                                cropped_imgs.append(cropped_img)

                    bboxes = np.asarray(bboxes)
                    cropped_imgs = np.asarray(cropped_imgs)
                    obj_repr["bbox"][view] = bboxes
                    obj_repr["cropped_img"][view] = cropped_imgs
                filled_prompt.append(obj_repr)
        raw_prompt = [filled_prompt]
        max_n_objs_prompt = {view: 0 for view in views}
        for prompt in raw_prompt:
            for token in prompt:
                if isinstance(token, dict):
                    for view in views:
                        max_n_objs_prompt[view] = max(
                            max_n_objs_prompt[view], len(
                                token["cropped_img"][view])
                        )
        raw_prompt_token_type, word_batch, image_batch = [], [], []
        for prompt in raw_prompt:
            token_type = []
            for token in prompt:
                if isinstance(token, int):
                    token_type.append(0)
                    word_batch.append(token)
                elif isinstance(token, dict):
                    token_type.append(1)
                    n_objs_prompt = {
                        view: len(token["cropped_img"][view]) for view in views
                    }
                    # add mask
                    token["mask"] = {
                        view: np.ones((n_objs_prompt[view],), dtype=bool)
                        for view in views
                    }
                    n_objs_to_pad = {
                        view: max_n_objs_prompt[view] - n_objs_prompt[view]
                        for view in views
                    }
                    objs_pad = {
                        "bbox": {
                            view: np.zeros(
                                (n_objs_to_pad[view], 4), dtype=np.int64)
                            for view in views
                        },
                        "cropped_img": {
                            view: np.zeros(
                                (n_objs_to_pad[view], 3, 32, 32),
                                dtype=np.uint8,
                            )
                            for view in views
                        },
                        "mask": {
                            view: np.zeros((n_objs_to_pad[view]), dtype=bool)
                            for view in views
                        },
                    }
                    token = any_concat([token, objs_pad], dim=0)
                    image_batch.append(token)
            raw_prompt_token_type.append(token_type)
        assert sum([len(prompt) for prompt in raw_prompt_token_type]) == len(
            word_batch) + len(image_batch)

        raw_prompt_token_type = np.array(raw_prompt_token_type[0])
        word_batch = any_stack(word_batch, dim=0)
        image_batch = any_to_datadict(stack_sequence_fields(image_batch))

        word_batch = any_to_torch_tensor(word_batch)
        image_batch = image_batch.to_torch_tensor()
        return raw_prompt_token_type, word_batch, image_batch

    def _prepare_obs(self, agent_traj: object, chosen_t: list, views: list, task_name: str, rgb_dict: dict = None):

        obs_list = {
            "ee": None,
            "objects": {
                "cropped_img": {view: [] for view in views},
                "bbox": {view: [] for view in views},
                "mask": {view: [] for view in views},
            },
        }

        for t in chosen_t:
            obs_t = agent_traj.get(t)

            obs_list['ee'] = torch.from_numpy(
                np.array([0]))  # obs_t['obs']['gripper_qpos']
            # for each view compute crop of the objects of interest
            for view in views:
                # get observation at timestamp t
                obs_t = agent_traj.get(t)['obs']
                rgb_this_view = obs_t['camera_front_image'][:, :, ::-1]
                # cv2.imwrite("rgb_this_view.png", np.array(rgb_this_view))
                bboxes = []
                cropped_imgs = []
                n_pad = 0

                # cut the image around each object in the scene
                for obj_name in ENV_OBJECTS[task_name]['obj_names']:
                    # get object bb
                    obj_bb = obs_t['obj_bb']['camera_front'][obj_name]
                    upper_left_corner = obj_bb['upper_left_corner']
                    bottom_right_corner = obj_bb['bottom_right_corner']
                    object_center = obj_bb['center']
                    # bounding box center, height and width
                    x_center, y_center = object_center[0], object_center[1]
                    h, w = upper_left_corner[1] - \
                        bottom_right_corner[1], upper_left_corner[0] - \
                        bottom_right_corner[0]
                    bboxes.append(
                        [int(x_center), int(y_center), int(h), int(w)])
                    # crop image
                    cropped_img = np.array(rgb_this_view[
                        bottom_right_corner[1]:upper_left_corner[1] + 1, bottom_right_corner[0]:upper_left_corner[0] + 1, :])
                    # cv2.imwrite("cropped_img_obs.png",
                    #             np.array(cropped_img))

                    # pad if dimensions are different
                    if cropped_img.shape[0] != cropped_img.shape[1]:
                        diff = abs(
                            cropped_img.shape[0] - cropped_img.shape[1])
                        pad_before, pad_after = int(
                            diff / 2), diff - int(diff / 2)
                        if cropped_img.shape[0] > cropped_img.shape[1]:
                            pad_width = (
                                (0, 0), (pad_before, pad_after), (0, 0))
                        else:
                            pad_width = (
                                (pad_before, pad_after), (0, 0), (0, 0))
                        cropped_img = np.pad(
                            cropped_img,
                            pad_width,
                            mode="constant",
                            constant_values=0,
                        )
                        assert cropped_img.shape[0] == cropped_img.shape[1], "INTERNAL"
                    cropped_img = np.asarray(cropped_img)
                    cropped_img = cv2.resize(
                        cropped_img,
                        (32, 32),
                        interpolation=cv2.INTER_AREA,
                    )
                    cropped_img = rearrange(cropped_img, "h w c -> c h w")
                    cropped_imgs.append(cropped_img)
                bboxes = np.asarray(bboxes)
                cropped_imgs = np.asarray(cropped_imgs)
                mask = np.ones(len(bboxes), dtype=bool)

                obs_list["objects"]["bbox"][view].append(bboxes)
                obs_list["objects"]["cropped_img"][view].append(cropped_imgs)
                obs_list["objects"]["mask"][view].append(mask)

        for view in views:
            obs_list["objects"]["bbox"][view] = np.stack(
                obs_list["objects"]["bbox"][view], axis=0
            )
            obs_list["objects"]["cropped_img"][view] = np.stack(
                obs_list["objects"]["cropped_img"][view], axis=0
            )
            obs_list["objects"]["mask"][view] = np.stack(
                obs_list["objects"]["mask"][view], axis=0
            )

        obs = any_to_datadict(obs_list)
        obs = obs.to_torch_tensor()
        # obs = any_transpose_first_two_axes(obs)

        return obs

    def _create_prompt_assets(self, obs: dict, task_name: str, views: list, modalities: list):
        prompt_assets = dict()
        prompt_assets['pick_object'] = dict()

        if task_name == 'pick_place' or task_name == 'nut_assembly':
            prompt_assets['pick_object']['rgb'] = dict()
            prompt_assets['pick_object']['segm'] = dict({'obj_info': dict()})
            prompt_assets['pick_object']['placeholder_type'] = 'object'
            # For each modality fill the prompt asset
            for modality in modalities:
                # For each modality and for each view fill the prompt asset
                for view in views:
                    if view not in prompt_assets['pick_object'][modality].keys():
                        prompt_assets['pick_object'][modality][view] = dict()
                    target_obj_id = obs['target-object']
                    target_obj_name = ENV_OBJECTS[task_name]['obj_names'][target_obj_id]
                    # assign prompt assets
                    prompt_assets['pick_object'][modality][view] = obs['camera_front_image'][:, :, ::-1]
                    prompt_assets['pick_object']['segm']['obj_info']['obj_id'] = target_obj_id
                    prompt_assets['pick_object']['segm']['obj_info']['obj_name'] = ENV_OBJECTS[task_name]['splitted_obj_names'][target_obj_id]
                    prompt_assets['pick_object']['segm']['obj_info']['obj_color'] = ENV_OBJECTS[task_name]['splitted_obj_names'][target_obj_id].split(" ")[
                        0]

        return prompt_assets


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


if __name__ == '__main__':
    pass
