import random
import torch
from os.path import join, expanduser
from multi_task_il.datasets import load_traj, split_files
import cv2
from torch.utils.data._utils.collate import default_collate
from robosuite.utils.transform_utils import quat2axisangle, axisangle2quat, quat2mat, mat2quat 
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
import time
import math
from PIL import Image
from tqdm import tqdm
import logging
import itertools
from torchvision.transforms import ToPILImage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')

# Create a logger object
logger = logging.getLogger('Data-Loader')

T_bl_sim_to_w_sim = np.array([[0, -1, 0, 0], 
                              [1, 0, 0, 0.612],
                              [0, 0, 1, -0.860],
                              [0, 0, 0, 1]])
R_g_sim_to_g_robot = np.array([[0, -1, 0], 
                              [1, 0, 0],
                              [0, 0, 1]])


DEBUG = False

OBJECTS_POS_DIM = {
    'pick_place': {
        'obj_names': ['greenbox', 'yellowbox', 'bluebox', 'redbox', 'bin'],
        'bin_position': [0.18, 0.00, 0.75],
        'obj_dim': {'greenbox': [0.05, 0.055, 0.045],  # W, H, D
                    'yellowbox': [0.05, 0.055, 0.045],
                    'bluebox': [0.05, 0.055, 0.045],
                    'redbox': [0.05, 0.055, 0.045],
                    'bin': [0.6, 0.06, 0.15],
                    'single_bin': [0.15, 0.06, 0.15]},
        'camera_names': {'camera_front', 'camera_lateral_right', 'camera_lateral_left'},
        'camera_pos':
            {
                'camera_front': [[0.45, -0.002826249197217832, 1.27]],
                'camera_lateral_left': [[-0.32693157973832665, 0.4625646268626449, 1.3]],
                'camera_lateral_right': [[-0.3582777207605626, -0.44377700364575223, 1.3]],
        },
        'camera_orientation':
            {
                'camera_front':  [0.6620018964346217, 0.26169506249574287, 0.25790267731943883, 0.6532651777140575],
                'camera_lateral_left': [-0.3050297127346233,  -0.11930536839029657, 0.3326804927221449, 0.884334095907446],
                'camera_lateral_right': [0.860369883903888, 0.3565444300005689, -0.1251454368177692, -0.3396500627826067],
        },
        'camera_fovy': 60,
        'img_dim': [200, 360]},

    'nut_assembly': {
        'obj_names': ['nut0', 'nut1', 'nut2'],
        'ranges': [[0.10, 0.31], [-0.10, 0.10], [-0.31, -0.10]]
    }
}

ENV_OBJECTS = {
    'pick_place': {
        'obj_names': ['greenbox', 'yellowbox', 'bluebox', 'redbox'],
        'ranges': [[-0.255, -0.195], [-0.105, -0.045], [0.045, 0.105], [0.195, 0.255]],
    },
    'nut_assembly': {
        'obj_names': ['nut0', 'nut1', 'nut2'],
        'ranges': [[0.10, 0.31], [-0.10, 0.10], [-0.31, -0.10]]
    },
    'stack_block': {
        'obj_names': ['cubeA', 'cubeB', 'cubeC'],
    },
    'button': {
        'obj_names': ['machine1_goal1', 'machine1_goal2', 'machine1_goal3',
                      'machine2_goal1', 'machine2_goal2', 'machine2_goal3',
                      'machine1_goal1_final', 'machine1_goal2_final', 'machine1_goal3_final',
                      'machine2_goal1_final', 'machine2_goal2_final', 'machine2_goal3_final',],
        'obj_names_to_id': {'machine1_goal1': 0,
                            'machine1_goal2': 1,
                            'machine1_goal3': 2,
                            'machine2_goal1': 3,
                            'machine2_goal2': 4,
                            'machine2_goal3': 5,
                            'machine1_goal1_final': 6,
                            'machine1_goal2_final': 7,
                            'machine1_goal3_final': 8,
                            'machine2_goal1_final': 9,
                            'machine2_goal2_final': 10,
                            'machine2_goal3_final': 11}
    }
}



#
NUM_VARIATION_PER_OBEJECT = {'pick_place': (4, 4),
                             'nut_assembly': (3, 3),
                             'button': (1, 6),
                             'press_button_close_after_reaching': (1, 6),
                             'stack_block': (2, 6)}

STACK_BLOCK_TASK_ID_SEQUENCE = {0: 'rgb',
                                1: 'rbg',
                                2: 'bgr',
                                3: 'brg',
                                4: 'grb',
                                5: 'gbr', }


def collate_by_task(batch):
    """ Use this for validation: groups data by task names to compute per-task losses """
    collate_time = time.time()
    per_task_data = defaultdict(list)
    start_batch = time.time()
    for b in batch:
        per_task_data[b['task_name']].append(
            {k: v for k, v in b.items() if k != 'task_name' and k != 'task_id'}
        )
    logger.debug(f"Batch time {time.time()-start_batch}")

    collate_time = time.time()
    for name, data in per_task_data.items():
        per_task_data[name] = default_collate(data)
    logger.debug(f"Collate time {time.time()-collate_time}")
    return per_task_data


def create_train_val_dict(dataset_loader=object, agent_name: str = "ur5e", demo_name: str = "panda", root_dir: str = "", task_spec=None, split: list = [0.9, 0.1], allow_train_skip: bool = False, allow_val_skip: bool = False, mix_variations: bool = False, mode='train', mix_sim_real=False):

    sample_indx = 0
    agent_file_cnt = 0
    demo_file_cnt = 0
    count = 0
    pair_cnt = 0
    validation_on_skipped_task = False

    for spec in task_spec:
        
        if mode == 'val' and len(spec.get('skip_ids', [])) != 0:
            validation_on_skipped_task = False

        name, date = spec.get('name', None), spec.get('date', None)
        assert name, 'need to specify the task name for data generated, for easier tracking'
        dataset_loader.agent_files[name] = dict()
        dataset_loader.demo_files[name] = dict()

        dataset_loader.object_distribution[name] = OrderedDict()
        dataset_loader.object_distribution_to_indx[name] = OrderedDict()

        if dataset_loader.mode == 'train':
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
        dataset_loader.subtask_to_idx[name] = defaultdict(list)
        dataset_loader.demo_subtask_to_idx[name] = defaultdict(list)
        
       
        for _id in range(spec.get('n_tasks')):

            # take demo file from no-skipped tasks
            if not validation_on_skipped_task:
                if _id in spec.get('skip_ids', []):
                    if (allow_train_skip and dataset_loader.mode == 'train') or (allow_val_skip and dataset_loader.mode == 'val'):
                        print(
                            'Warning! Excluding subtask id {} from loaded **{}** dataset for task {}'.format(_id, dataset_loader.mode, name))
                        continue
            else:
                # take demo file from skipped tasks
                if _id not in spec.get('skip_ids', []):
                    if (allow_train_skip and dataset_loader.mode == 'train') or (allow_val_skip and dataset_loader.mode == 'val'):
                        print(
                            'Warning! Excluding subtask id {} from loaded **{}** dataset for task {}'.format(_id, dataset_loader.mode, name))
                        continue

            task_id = 'task_{:02d}'.format(_id)
            task_dir = expanduser(join(agent_dir,  task_id, '*.pkl'))
            agent_files = sorted(glob.glob(task_dir))
            
            if 'real' in task_dir and dataset_loader._mix_sim_real:
                task_dir_sim = task_dir.replace(agent_name, agent_name.replace('real_new_', ''))
                agent_files.extend(sorted(glob.glob(task_dir_sim)))
            if 'real' in task_dir and dataset_loader._dagger:
                dagger_path = join(task_dir.split('/opt_dataset/')[0], 'dagger_elaborated', task_dir.split('/opt_dataset/')[1])
                dagger_files = sorted(glob.glob(dagger_path))
                agent_files.extend(dagger_files)
                
            if len(agent_files) < 100: #and 'real' not in task_dir:
                agent_files = list(itertools.chain.from_iterable((e, e) for e in agent_files))
            
            assert len(agent_files) != 0, "Can't find dataset for task {}, subtask {} in dir {}".format(
                name, _id, task_dir)
            subtask_size = spec.get('traj_per_subtask', 100)
            if dataset_loader._dagger:
                subtask_size = len(agent_files)
            
            assert len(
                agent_files) >= subtask_size, "Doesn't have enough data "+str(len(agent_files))
            agent_files = agent_files[:subtask_size]

            # prev. version does split randomly, here we strictly split each subtask in the same split ratio:
            idxs = split_files(len(agent_files), split, dataset_loader.mode)
            agent_files = [agent_files[i] for i in idxs]

            task_dir = expanduser(join(demo_dir, task_id, '*.pkl'))

            demo_files = sorted(glob.glob(task_dir))

            subtask_size = spec.get('demo_per_subtask', 100)
            assert len(
                demo_files) >= subtask_size, "Doesn't have enough data "+str(len(demo_files))
            demo_files = demo_files[:subtask_size]
            idxs = split_files(len(demo_files), split, dataset_loader.mode)
            demo_files = [demo_files[i] for i in idxs]
            # assert len(agent_files) == len(demo_files), \
            #     'data for task {}, subtask #{} is not matched'.format(name, task_id)

            dataset_loader.agent_files[name][_id] = deepcopy(agent_files)
            dataset_loader.demo_files[name][_id] = deepcopy(demo_files)
            
            if not dataset_loader._mix_demo_agent:
                print(f"Loading {name} - {task_id}")
                for indx_demo, demo in enumerate(demo_files):
                    dataset_loader.all_demo_files[demo_file_cnt] = (name, _id, demo)
                    dataset_loader.demo_task_to_idx[name].append(demo_file_cnt)
                    dataset_loader.demo_subtask_to_idx[name][task_id].append(demo_file_cnt)
                    demo_file_cnt += 1
                    
                for indx_agent, agent in enumerate(agent_files):
                    # open file and check trajectory lenght
                    with open(agent, "rb") as f:
                        agent_data = pkl.load(f)
                        trj_len = agent_data['len']
                        
                        dataset_loader.all_agent_files[agent_file_cnt] = (name, _id, agent, trj_len)
                        agent_file_cnt += 1
                        
                        
                        # dataset_loader.all_file_pairs[sample_indx] = (name, _id, demo, agent, trj_len)
                        dataset_loader.task_to_idx[name].append(sample_indx)
                        dataset_loader.subtask_to_idx[name][task_id].append(sample_indx)
                        sample_indx += 1
                        count += trj_len
                        # pair_cnt += 1
                    

        # for C(T)OD                    
        if dataset_loader._mix_demo_agent:
            count = 0
            num_variation_per_object = NUM_VARIATION_PER_OBEJECT[name][0]
            num_objects = NUM_VARIATION_PER_OBEJECT[name][1]

            # for each sub-task
            for _id in range(spec.get('n_tasks')):
                # take demo file from no-skipped tasks
                if not validation_on_skipped_task:
                    if _id in spec.get('skip_ids', []):
                        if (allow_train_skip and dataset_loader.mode == 'train') or (allow_val_skip and dataset_loader.mode == 'val'):
                            print(
                                'Warning! Excluding subtask id {} from loaded **{}** dataset for task {}'.format(_id, dataset_loader.mode, name))
                            continue
                else:
                    # take demo file from skipped tasks
                    if _id not in spec.get('skip_ids', []):
                        if (allow_train_skip and dataset_loader.mode == 'train') or (allow_val_skip and dataset_loader.mode == 'val'):
                            print(
                                'Warning! Excluding subtask id {} from loaded **{}** dataset for task {}'.format(_id, dataset_loader.mode, name))
                            continue

                # for each demo_file
                demo_files = dataset_loader.demo_files[name][_id]
                for demo_file in demo_files:
                    # 50% trajectories same task
                    # 50% trajectories different files
                    same_variation_number = len(
                        dataset_loader.agent_files[name][_id])
                    # take the trajectories same variation as demo
                    if 'stack_block' in name:
                        same_sample_number = int(0.7*same_variation_number)
                        different_sample_number = same_variation_number-same_sample_number
                    else:
                        same_sample_number = int(0.5*same_variation_number)
                        different_sample_number = same_variation_number-same_sample_number
                    agent_files = random.sample(
                        dataset_loader.agent_files[name][_id], int(same_sample_number))
                    # take indices for different manipulated objects
                    target_obj_id = int(_id/num_variation_per_object)
                    for sub_task_id in range(spec.get('n_tasks')):
                        if not validation_on_skipped_task:
                            if sub_task_id in spec.get('skip_ids', []):
                                # print(f"Sub_task id {sub_task_id}")
                                continue
                        else:
                            if sub_task_id not in spec.get('skip_ids', []):
                                # print(f"Sub_task id {sub_task_id}")
                                continue

                        if sub_task_id == _id:
                            continue

                        if len(spec.get('skip_ids', [])) == 0:
                            if not (sub_task_id >= target_obj_id*num_variation_per_object and sub_task_id < ((target_obj_id*num_variation_per_object)+num_variation_per_object)):
                                # the following index has a differnt object
                                agent_files.extend(random.sample(
                                    dataset_loader.agent_files[name][sub_task_id], round(different_sample_number/(spec.get('n_tasks')-num_variation_per_object))))
                        else:
                            if not validation_on_skipped_task:
                                div = spec.get('n_tasks') - \
                                    len(spec.get('skip_ids', [])) - 1
                            else:
                                div = len(spec.get('skip_ids', [])) - 1
                            agent_files.extend(random.sample(
                                dataset_loader.agent_files[name][sub_task_id], round(different_sample_number / div)))
                    for agent_file in agent_files:
                        dataset_loader.all_file_pairs[count] = (
                            name, _id, demo_file, agent_file)
                        dataset_loader.task_to_idx[name].append(count)
                        dataset_loader.subtask_to_idx[name][_id].append(
                            count)
                        count += 1
                        pair_cnt += 1

        print(f"Task {name} has\n\tDemo files: {demo_file_cnt}\n\tAgent files: {agent_file_cnt}\n\tTotal pairs: {pair_cnt}\n\tTotal samples: {count}")

        if spec.get('demo_crop', None) is not None:
            dataset_loader.demo_crop[name] = spec.get(
                'demo_crop', [0, 0, 0, 0])
        if spec.get('agent_crop', None) is not None:
            dataset_loader.agent_crop[name] = spec.get(
                'agent_crop', [0, 0, 0, 0])
            
        if spec.get('agent_sim_crop', None) is not None and hasattr(dataset_loader, 'agent_sim_crop'):
            dataset_loader.agent_sim_crop[name] = spec.get(
                'agent_sim_crop', [0, 0, 0, 0])
            
        if spec.get('crop', None) is not None:
            dataset_loader.task_crops[name] = spec.get(
                'crop', [0, 0, 0, 0])

    return count, pair_cnt


def make_demo(dataset, traj, task_name, human_demo=False):
    """
    Do a near-uniform sampling of the demonstration trajectory
    """
    if dataset.select_random_frames:
        def clip(x): return int(max(1, min(x, len(traj) - 1)))
        per_bracket = max(len(traj) / dataset._demo_T, 1)
        frames = []
        cp_frames = []
        for i in range(dataset._demo_T):
            # fix to using uniform + 'sample_side' now
            if i == dataset._demo_T - 1:
                n = len(traj) - 1
            elif i == 0:
                n = 1
            else:
                n = clip(np.random.randint(
                    int(i * per_bracket), int((i + 1) * per_bracket)))
            # frames.append(_make_frame(n))

            if not human_demo and (dataset.width != 224 and dataset.height != 224):
                obs = copy.copy(
                    traj.get(n)['obs']['camera_front_image'][:, :, ::-1])
            elif dataset.width == 224 and dataset.height == 224:
                obs = copy.copy(
                    traj.get(n)['obs']['camera_front_image'])
            else:
                obs = copy.copy(
                    traj.get(n)['obs']['camera_front_image'])
            
            processed = dataset.frame_aug(
                task_name,
                obs,
                perform_aug=False,
                frame_number=i,
                perform_scale_resize=True)
            frames.append(processed)
            if dataset.aug_twice:
                cp_frames.append(dataset.frame_aug(
                    task_name,
                    obs,
                    True,
                    perform_aug=False,
                    perform_scale_resize=True))
    else:
        frames = []
        cp_frames = []
        for i in range(dataset._demo_T):
            # get first frame
            if i == 0:
                n = 1
            # get the last frame
            elif i == dataset._demo_T - 1:
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

            processed = dataset.frame_aug(task_name,
                                          obs,
                                          perform_aug=False,
                                          perform_scale_resize=True,
                                          agent=False)
            frames.append(processed)
            if dataset.aug_twice:
                cp_frames.append(dataset.frame_aug(
                    task_name,
                    obs,
                    True,
                    perform_aug=False,
                    perform_scale_resize=True))

    ret_dict = dict()
    ret_dict['demo'] = torch.stack(frames)
    if dataset.aug_twice:
        ret_dict['demo_cp'] = torch.stack(cp_frames)
    return ret_dict


def adjust_bb(dataset_loader, bb, obs, img_width=360, img_height=200, top=0, left=0, box_w=360, box_h=200):
    # For each bounding box
    bb = np.array(bb)
    if len(bb.shape) == 3:
        bb = bb[:, 0, :]
    for obj_indx, obj_bb in enumerate(bb):
        if len(obj_bb.shape) == 2:
            obj_bb = obj_bb[0]
        # Convert normalized bounding box coordinates to actual coordinates
        x1_old, y1_old, x2_old, y2_old = obj_bb
        x1_old = int(x1_old)
        y1_old = int(y1_old)
        x2_old = int(x2_old)
        y2_old = int(y2_old)

        # Modify bb based on computed resized-crop
        # 1. Take into account crop and resize
        x_scale = dataset_loader.width/box_w
        y_scale = dataset_loader.height/box_h
        x1 = int((x1_old - left) * x_scale)
        x2 = int((x2_old - left) * x_scale)
        y1 = int((y1_old - top) * y_scale)
        y2 = int((y2_old - top) * y_scale)

        if x1 <= 0:
            x1 = 0
        if x2 <= 0:
            x2 = 0
        if y1 <= 0:
            y1 = 0
        if y2 <= 0:
            y2 = 0

        if x1 >= dataset_loader.width:
            x1 = dataset_loader.width-1
        if x2 >= dataset_loader.width:
            x2 = dataset_loader.width-1
        if y1 >= dataset_loader.height:
            y1 = dataset_loader.height-1
        if y2 >= dataset_loader.height:
            y2 = dataset_loader.height-1

        if DEBUG:
            image = cv2.rectangle(np.ascontiguousarray(np.array(np.moveaxis(
                obs.numpy()*255, 0, -1), dtype=np.uint8)),
                (x1,
                    y1),
                (x2,
                    y2),
                color=(0, 0, 255),
                thickness=1)
            
            cv2.imwrite("bb_cropped.png", image)    

        # replace with new bb
        bb[obj_indx] = np.array([[x1, y1, x2, y2]])
    return bb


def create_gt_bb(dataset_loader, traj, step_t, task_name, distractor=False, command=None, subtask_id=-1, agent_task_id=-1, take_place_loc=False):
    bb = []
    cl = []

    if subtask_id == -1:
        # 1. Get Target Object
        if task_name != "stack_block":
            if "button" not in task_name:
                if step_t['obs'].get('target-object', None) is None:
                    target_obj_id = int(agent_task_id /
                                        NUM_VARIATION_PER_OBEJECT[task_name][1])
                else:
                    target_obj_id = step_t['obs']['target-object']
            else:
                target_obj_id = ENV_OBJECTS['button']['obj_names_to_id'][step_t['obs']
                                                                         ['target-object']]
        else:
            agent_target = 0
    else:
        if task_name != "stack_block":
            target_obj_id = int(
                subtask_id/NUM_VARIATION_PER_OBEJECT[task_name][0])
            target_place_id = int(subtask_id %
                                  NUM_VARIATION_PER_OBEJECT[task_name][0]) if 'button' not in task_name else target_obj_id+NUM_VARIATION_PER_OBEJECT[task_name][1]
        else:
            demo_target = STACK_BLOCK_TASK_ID_SEQUENCE[subtask_id][0]
            demo_place = STACK_BLOCK_TASK_ID_SEQUENCE[subtask_id][1]
            agent_target = STACK_BLOCK_TASK_ID_SEQUENCE[agent_task_id].find(
                demo_target)
            agent_place = STACK_BLOCK_TASK_ID_SEQUENCE[agent_task_id].find(
                demo_place)
        # if target_obj_id != step_t['obs']['target-object']:
        #     print("different-objects")

    if task_name == 'pick_place':
        num_objects = 3
    elif task_name == 'nut_assembly':
        num_objects = 2
    elif 'button' in task_name:
        num_objects = 5
        target_obj_id = ENV_OBJECTS['button']['obj_names_to_id'][ENV_OBJECTS['button']
                                                                 ['obj_names'][target_obj_id]]
        if take_place_loc:
            target_place_id = ENV_OBJECTS['button']['obj_names_to_id'][ENV_OBJECTS['button']
                                                                       ['obj_names'][target_place_id]]
    if task_name != "stack_block":
        # select randomly another object
        no_target_obj_id = target_obj_id
        while no_target_obj_id == target_obj_id:
            no_target_obj_id = random.randint(
                0, num_objects)
        if take_place_loc:
            # select randomly another place
            no_target_place_id = target_place_id
            while no_target_place_id == target_place_id:
                if 'button' not in task_name:
                    no_target_place_id = random.randint(
                        0, num_objects)
                else:
                    no_target_place_id = random.randint(
                        num_objects+1, (2*num_objects)+1)
    else:
        obj_list = ["cubeA", "cubeB", "cubeC"]
        agent_target_name = obj_list.pop(agent_target)
        no_target_obj_name = random.choice(obj_list)
        if take_place_loc:
            obj_list = ["cubeA", "cubeB", "cubeC"]
            agent_place_name = obj_list.pop(agent_place)
            no_place_obj_name = random.choice(obj_list)
    try:
        dict_keys = list(step_t['obs']['obj_bb']['camera_front'].keys())
        if take_place_loc and 'button' not in task_name and task_name != 'stack_block':
            target_place_id = target_place_id + \
                NUM_VARIATION_PER_OBEJECT[task_name][0] + \
                1*(task_name == 'pick_place' and 'bin' in dict_keys)
            no_target_place_id = no_target_place_id + \
                NUM_VARIATION_PER_OBEJECT[task_name][0] + \
                1*(task_name == 'pick_place' and 'bin' in dict_keys)
    except:
        dict_keys = list(step_t['obs']['obj_bb'].keys())

    if distractor:
        end = 4 if take_place_loc else 2
    else:
        end = 1

    for i in range(end):
        if task_name != "stack_block":
            if i == 0:
                object_name = dict_keys[target_obj_id]
            elif i == 1 and distractor:
                object_name = dict_keys[no_target_obj_id]
            elif i == 2 and distractor:
                object_name = dict_keys[target_place_id]
            elif i == 3 and distractor:
                object_name = dict_keys[no_target_place_id]
        else:
            if i == 0:
                object_name = agent_target_name
            elif i == 0 and distractor:
                object_name = no_target_obj_name
            elif i == 2 and distractor:
                object_name = agent_place_name
            elif i == 3 and distractor:
                object_name = no_place_obj_name
        try:
            # if not getattr(dataset_loader, "real", False):
            top_left = step_t['obs']['obj_bb']["camera_front"][object_name]['bottom_right_corner']
            bottom_right = step_t['obs']['obj_bb']["camera_front"][object_name]['upper_left_corner']
            # else:
            #     top_left = step_t['obs']['obj_bb']["camera_front"][object_name]['upper_left_corner']
            #     bottom_right = step_t['obs']['obj_bb']["camera_front"][object_name]['bottom_right_corner']
        except:
            top_left = step_t['obs']['obj_bb'][object_name]['upper_left_corner']
            bottom_right = step_t['obs']['obj_bb'][object_name]['bottom_right_corner']

        # check bb coordinates
        assert top_left[0] < bottom_right[0] and top_left[1] < bottom_right[1], "Error on bb"
        # 2. Get stored BB
        top_left_x = top_left[0]
        top_left_y = top_left[1]
        # print(f"Top-Left X {top_left_x} - Top-Left Y {top_left_y}")
        bottom_right_x = bottom_right[0]
        bottom_right_y = bottom_right[1]

        # test GT
        if DEBUG:
            if i == 0 or i == 2:
                color = (0, 255, 0)
                image = np.array(
                    step_t['obs']['camera_front_image'][:, :, ::-1])
            else:
                color = (255, 0, 0)
            image = cv2.rectangle(image,
                                  (int(top_left_x),
                                   int(top_left_y)),
                                  (int(bottom_right_x),
                                   int(bottom_right_y)),
                                  color=color,
                                  thickness=1)
            if DEBUG:
                cv2.imwrite("GT_bb_prova.png", image)

        bb.append([top_left_x, top_left_y,
                   bottom_right_x, bottom_right_y])

        # 1 Target
        # 0 No-target
        # 2 Target-place
        # 3 No-Target-place

        if i == 0:
            cl.append(1)
        elif i == 1:
            cl.append(0)
        elif i == 2:
            cl.append(2)
        elif i == 3:
            cl.append(3)

    if DEBUG:
        image = np.array(
            step_t['obs']['camera_front_image'][:, :, ::-1])
        for i, single_bb in enumerate(bb):
            if i == 0 or i == 3:
                color = (0, 255, 0) # green no-targ
            else:
                color = (255, 0, 0) # blue target
            image = cv2.rectangle(image,
                                  (int(single_bb[0]),
                                   int(single_bb[1])),
                                  (int(single_bb[2]),
                                   int(single_bb[3])),
                                  color=color,
                                  thickness=1)
        cv2.imwrite("GT_bb_prova_full_bb.png", image)

    return np.array(bb), np.array(cl)


def create_gt_bb_all_obj(dataset_loader, traj, step_t, task_name, distractor=False, command=None, subtask_id=-1, agent_task_id=-1, take_place_loc=False):
    bb = []
    cl = []

    if subtask_id == -1:
        # 1. Get Target Object
        if task_name != "stack_block":
            if task_name != "button":
                target_obj_id = step_t['obs']['target-object']
            else:
                target_obj_id = ENV_OBJECTS['button']['obj_names_to_id'][step_t['obs']
                                                                         ['target-object']]
        else:
            agent_target = 0
    else:
        if task_name != "stack_block":
            target_obj_id = int(
                subtask_id/NUM_VARIATION_PER_OBEJECT[task_name][0])
            target_place_id = int(subtask_id %
                                  NUM_VARIATION_PER_OBEJECT[task_name][0])
        else:
            demo_target = STACK_BLOCK_TASK_ID_SEQUENCE[subtask_id][0]
            agent_target = STACK_BLOCK_TASK_ID_SEQUENCE[agent_task_id].find(
                demo_target)
        # if target_obj_id != step_t['obs']['target-object']:
        #     print("different-objects")

    if task_name == 'pick_place':
        num_objects = 3
    elif task_name == 'nut_assembly':
        num_objects = 2
    elif task_name == 'button':
        num_objects = 5
        target_obj_id = ENV_OBJECTS['button']['obj_names_to_id'][ENV_OBJECTS['button']
                                                                 ['obj_names'][target_obj_id]]
    if task_name != "stack_block":
        # select randomly another object
        no_target_obj_id = target_obj_id
        while no_target_obj_id == target_obj_id:
            no_target_obj_id = random.randint(
                0, num_objects)
        if take_place_loc:
            # select randomly another place
            no_target_place_id = target_place_id
            while no_target_place_id == target_place_id:
                no_target_place_id = random.randint(
                    0, num_objects)
    else:
        obj_list = ["cubeA", "cubeB", "cubeC"]
        agent_target_name = obj_list.pop(agent_target)
        no_target_obj_name = random.choice(obj_list)

    try:
        dict_keys = list(step_t['obs']['obj_bb']['camera_front'].keys())
        if take_place_loc:
            target_place_id = target_place_id + \
                NUM_VARIATION_PER_OBEJECT[task_name][0] + \
                1*(task_name == 'pick_place')
            no_target_place_id = no_target_place_id + \
                NUM_VARIATION_PER_OBEJECT[task_name][0] + \
                1*(task_name == 'pick_place')
    except:
        dict_keys = list(step_t['obs']['obj_bb'].keys())

    if distractor:
        end = 2 * (2*take_place_loc)
    else:
        end = 1

    for i in range(end):
        if task_name != "stack_block":
            if i == 0:
                object_name = dict_keys[target_obj_id]
            elif i == 1 and distractor:
                object_name = dict_keys[no_target_obj_id]
            elif i == 2 and distractor:
                object_name = dict_keys[target_place_id]
            elif i == 3 and distractor:
                object_name = dict_keys[no_target_place_id]
        else:
            if i == 0:
                object_name = agent_target_name
            elif i != 0 and distractor:
                object_name = no_target_obj_name

        try:
            if not getattr(dataset_loader, "real", False):
                top_left = step_t['obs']['obj_bb']["camera_front"][object_name]['bottom_right_corner']
                bottom_right = step_t['obs']['obj_bb']["camera_front"][object_name]['upper_left_corner']
            else:
                top_left = step_t['obs']['obj_bb']["camera_front"][object_name]['upper_left_corner']
                bottom_right = step_t['obs']['obj_bb']["camera_front"][object_name]['bottom_right_corner']
        except:
            top_left = step_t['obs']['obj_bb'][object_name]['upper_left_corner']
            bottom_right = step_t['obs']['obj_bb'][object_name]['bottom_right_corner']

        # 2. Get stored BB
        top_left_x = top_left[0]
        top_left_y = top_left[1]
        # print(f"Top-Left X {top_left_x} - Top-Left Y {top_left_y}")
        bottom_right_x = bottom_right[0]
        bottom_right_y = bottom_right[1]

        # test GT
        if DEBUG:
            if i == 0 or i == 2:
                color = (0, 255, 0)
                image = np.array(
                    step_t['obs']['camera_front_image'][:, :, ::-1])
            else:
                color = (255, 0, 0)
            image = cv2.rectangle(image,
                                  (int(top_left_x),
                                   int(top_left_y)),
                                  (int(bottom_right_x),
                                   int(bottom_right_y)),
                                  color=color,
                                  thickness=1)
            if DEBUG:
                cv2.imwrite("GT_bb_prova.png", image)

        bb.append([top_left_x, top_left_y,
                   bottom_right_x, bottom_right_y])

        # 1 Target
        # 0 No-target
        # 2 Target-plase
        # 3 No-Target-place

        if i == 0:
            cl.append(1)
        elif i == 1:
            cl.append(0)
        elif i == 2:
            cl.append(2)
        elif i == 3:
            cl.append(3)

    return np.array(bb), np.array(cl)


def create_gt_bb_sequence(dataset_loader, traj, t, task_name, distractor, command, subtask_id, agent_task_id):
    # Create a sequence of T bbs, starting from t and ending with T
    bb_list = list()
    cl_list = list()
    for current_t in range(t, t+dataset_loader._bbs_T):
        bb_t, cl_t = create_gt_bb(dataset_loader=dataset_loader,
                                  traj=traj,
                                  step_t=traj.get(current_t),
                                  task_name=task_name,
                                  distractor=distractor,
                                  command=command,
                                  subtask_id=subtask_id,
                                  agent_task_id=agent_task_id)
        bb_list.append(bb_t)
        cl_list.append(cl_t)
    return np.array(bb_t), np.array(cl_list)


def adjust_points(points, frame_dims, crop_params, height, width):

    h = np.clip(points[0] - crop_params[0], 0,
                frame_dims[0] - crop_params[1])
    w = np.clip(points[1] - crop_params[2], 0,
                frame_dims[1] - crop_params[3])
    h = float(
        h) / (frame_dims[0] - crop_params[0] - crop_params[1]) * height
    w = float(
        w) / (frame_dims[1] - crop_params[2] - crop_params[3]) * width
    return tuple([int(min(x, d - 1)) for x, d in zip([h, w], (height, width))])


def trasform_from_world_to_bl(action):
    aa_gripper = action[3:-1]
    # convert axes-angle into rotation matrix
    R_w_sim_to_gripper_sim = quat2mat(axisangle2quat(aa_gripper))
    
    gripper_pos = action[0:3]
    
    T_w_sim_gripper_sim = np.zeros((4,4))
    T_w_sim_gripper_sim[3,3] = 1
    
    # position
    T_w_sim_gripper_sim[0,3] = gripper_pos[0]
    T_w_sim_gripper_sim[1,3] = gripper_pos[1]
    T_w_sim_gripper_sim[2,3] = gripper_pos[2]
    # orientation
    T_w_sim_gripper_sim[0:3, 0:3] = R_w_sim_to_gripper_sim
    
    T_bl_sim_gripper_sim = T_bl_sim_to_w_sim @ T_w_sim_gripper_sim
    
    # print(f"Transformation from world to bl:\n{T_bl_sim_gripper_sim}")
    
    R_bl_to_gripper_sim = T_bl_sim_gripper_sim[0:3, 0:3]
    
    R_bl_to_gripper_real = R_bl_to_gripper_sim @ R_g_sim_to_g_robot
    
    action_bl = np.zeros((7))
    action_bl[0:3] = T_bl_sim_gripper_sim[0:3, 3]
    action_bl[3:6] = quat2axisangle(mat2quat(R_bl_to_gripper_real))
    if action[-1] == -1:
        action_bl[6] = 0
    else:
        action_bl[6] = 1
    
    return action_bl

def create_sample(dataset_loader, traj, chosen_t, task_name, command, load_action=False, load_state=False, load_eef_point=False, distractor=False, subtask_id=-1, agent_task_id=-1, bb_sequence=False, take_place_loc=False, sim_crop=True, convert_action=True, human_demo=False):

    images = []
    images_cp = []
    bb = []
    obj_classes = []
    actions = []
    states = []
    points = []

    time_sample = time.time()
    crop_params = dataset_loader.task_crops.get(task_name, [0, 0, 0, 0])
    for j, t in enumerate(chosen_t):
        try:
            t = t.item()
        except:
            pass
        step_t = traj.get(t)

        if not getattr(dataset_loader, "real", False) or (getattr(dataset_loader, "real", False) and sim_crop):
            if not human_demo and (dataset_loader.width != 224 and dataset_loader.height != 224):
                image = copy.copy(
                    step_t['obs']['camera_front_image'][:, :, ::-1])
            elif not human_demo and (dataset_loader.width == 224 and dataset_loader.height == 224):
                image = copy.copy(
                    step_t['obs']['camera_front_image'])
            else:
                image = copy.copy(
                    step_t['obs']['camera_front_image'])
        else:
            if step_t['obs'].get('camera_front_image_full_size', None) is not None:
                image = copy.copy(
                cv2.imdecode(step_t['obs']['camera_front_image_full_size'], cv2.IMREAD_COLOR))
            else:
                if not human_demo and (dataset_loader.width != 224 and dataset_loader.height != 224):
                    image = copy.copy(
                        step_t['obs']['camera_front_image'])
                elif not human_demo and (dataset_loader.width == 224 and dataset_loader.height == 224):
                    image = copy.copy(
                        step_t['obs']['camera_front_image'][:,:,::-1])
                else:
                    image = copy.copy(
                        step_t['obs']['camera_front_image'][:,:,::-1])

        if DEBUG:
            cv2.imwrite("original_image.png", image)

        # Create GT BB
        bb_time = time.time()
        if getattr(dataset_loader, '_bbs_T', 1) == 1:
            bb_frame, class_frame = create_gt_bb(dataset_loader=dataset_loader,
                                                 traj=traj,
                                                 step_t=step_t,
                                                 task_name=task_name,
                                                 distractor=distractor,
                                                 command=command,
                                                 subtask_id=subtask_id,
                                                 agent_task_id=agent_task_id,
                                                 take_place_loc=take_place_loc)

            logger.debug(f"BB time {time.time()-bb_time}")
        else:
            bb_frame, class_frame = create_gt_bb_sequence(dataset_loader=dataset_loader,
                                                          traj=traj,
                                                          t=t,
                                                          task_name=task_name,
                                                          distractor=distractor,
                                                          command=command,
                                                          subtask_id=subtask_id,
                                                          agent_task_id=agent_task_id)
        # print(f"BB time: {end_bb-start_bb}")

        if dataset_loader._perform_augs:
            # Append bb, obj classes and images
            aug_time = time.time()
            processed, bb_aug, class_frame = dataset_loader.frame_aug(
                task_name,
                image,
                False,
                bb_frame,
                class_frame,
                perform_scale_resize=getattr(
                    dataset_loader, "perform_scale_resize", True),
                agent=True,
                sim_crop=sim_crop)
            end_aug = time.time()
            logger.debug(f"Aug time: {end_aug-aug_time}")
            images.append(processed)
            
            # pil_image = ToPILImage()(copy.deepcopy(processed).cpu())
            # pil_image.save("augmented_image.png")
            
            
        else:
            bb_aug = bb_frame

        bb.append(torch.from_numpy(bb_aug.astype(np.int32)))
        obj_classes.append((torch.from_numpy(class_frame.astype(np.int32))))

        if dataset_loader.aug_twice:
            aug_twice_time = time.time()
            image_cp = dataset_loader.frame_aug(task_name,
                                                image,
                                                True,
                                                perform_scale_resize=getattr(
                                                    dataset_loader, "perform_scale_resize", True),
                                                agent=True,
                                                sim_crop=sim_crop)
            images_cp.append(image_cp)
            logger.debug(f"Aug twice time: {time.time()-aug_twice_time}")

        if load_eef_point:
            eef_point_time = time.time()
            if DEBUG:
                image_point = np.array(
                    step_t['obs']['camera_front_image'][:, :, ::-1], dtype=np.uint8)
                image_point = cv2.circle(cv2.UMat(image_point), (step_t['obs']['eef_point'][1], step_t['obs']['eef_point'][0]), radius=1, color=(
                    0, 0, 255), thickness=1)
                cv2.imwrite("gt_point.png", cv2.UMat(image_point))

            points.append(np.array(
                adjust_points(step_t['obs']['eef_point'],
                              image.shape[:2],
                              crop_params=crop_params,
                              height=dataset_loader.height,
                              width=dataset_loader.width)))

            if DEBUG:
                image = np.array(np.moveaxis(
                    processed.numpy()*255, 0, -1), dtype=np.uint8)
                image = cv2.circle(cv2.UMat(image), (points[-1][0][1], points[-1][0][0]), radius=1, color=(
                    0, 0, 255), thickness=1)
                cv2.imwrite("adjusted_point.png", cv2.UMat(image))
            logger.debug(f"EEF point: {time.time()-eef_point_time}")

        if load_action and (j >= 1 or ("real" in dataset_loader.agent_name and not dataset_loader.pick_next)):
            action_time = time.time()
            # Load action
            action_list = list()
            for next_t in range(dataset_loader._action_T):
                if t+next_t <= len(traj)-1:
                    action = step_t['action'] if next_t == 0 else traj.get(
                        t+next_t)['action']
                else:
                    action = step_t['action']
                if "real" in dataset_loader.agent_name:
                    if not sim_crop and action.shape[0] == 8:
                        from robosuite.utils.transform_utils import quat2axisangle
                        rot_quat = action[3:7]
                        rot_axis_angle = quat2axisangle(rot_quat)
                        action = normalize_action(
                            action=np.concatenate(
                                (action[:3], rot_axis_angle, [action[7]])),
                            n_action_bin=dataset_loader._n_action_bin,
                            action_ranges=dataset_loader._normalization_ranges)
                    elif not sim_crop and action.shape[0] == 7:
                        action =normalize_action(
                            action=action,
                            n_action_bin=dataset_loader._n_action_bin,
                            action_ranges=dataset_loader._normalization_ranges)   
                    else:
                        action =normalize_action(
                            action=trasform_from_world_to_bl(action),
                            n_action_bin=dataset_loader._n_action_bin,
                            action_ranges=dataset_loader._normalization_ranges)                 
                else:
                    if dataset_loader._normalize_action:
                        if not convert_action:
                            action = normalize_action(
                                action=action,
                                n_action_bin=dataset_loader._n_action_bin,
                                action_ranges=dataset_loader._normalization_ranges)
                        else:
                            action = normalize_action(
                                action=trasform_from_world_to_bl(action),
                                n_action_bin=dataset_loader._n_action_bin,
                                action_ranges=dataset_loader._normalization_ranges)
                action_list.append(action)

            actions.append(action_list)
            logger.debug(f"Action: {time.time()-action_time}")

        if load_state:
            state_time = time.time()
            state = []
            # Load states
            for k in dataset_loader._state_spec:
                if k == 'action':
                    norm_start = time.time()
                    state_component = normalize_action(
                        action=step_t['action'],
                        n_action_bin=dataset_loader._n_action_bin,
                        action_ranges=dataset_loader._normalization_ranges)
                    norm_end = time.time()
                    # print(f"Norm time {norm_end-norm_start}")
                elif k == 'gripper_state':
                    try:
                        state_component = np.array(
                            [step_t['action'][-1]], dtype=np.float32)
                    except:
                        print("Error on gripper state")
                else:                        
                    if step_t['obs'].get(k, None) is not None and isinstance(step_t['obs'][k], int):
                        state_component = np.array(
                            [step_t['obs'][k]], dtype=np.float32)
                    else:
                        if "real" in dataset_loader.agent_name:
                            if step_t['obs'].get(k, None) is not None and k == 'joint_pos':
                                state_component = np.array(
                                    step_t['obs'][k], dtype=np.float32)
                            elif step_t['obs'].get(k, None) is None and k == 'joint_pos':
                                state_component = step_t['obs']['state'][:-1]
                            shoulder_pan_joint = state_component[2]
                            state_component[2] = state_component[0]
                            state_component[0] = shoulder_pan_joint
                        else:
                            state_component = np.array(
                                step_t['obs'][k], dtype=np.float32)
                            
                        
                state.append(state_component)
            states.append(np.concatenate(state))
            logger.debug(f"State: {time.time()-state_time}")

        # test GT
        if DEBUG:
            image = np.array(np.moveaxis(
                images[j].cpu().numpy()*255, 0, -1), dtype=np.uint8)
            image = cv2.rectangle(np.ascontiguousarray(image),
                                  (int(bb_aug[0][0]),
                                   int(bb_aug[0][1])),
                                  (int(bb_aug[0][2]),
                                   int(bb_aug[0][3])),
                                  color=(0, 0, 255),
                                  thickness=1)
            # print(f"Command {command}")
            cv2.imwrite("GT_bb_after_aug.png", image)
    end_time_sample = time.time()
    logger.debug(f"Sample time {end_time_sample-time_sample}")
    return images, images_cp, bb, obj_classes, actions, states, points


