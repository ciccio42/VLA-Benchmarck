import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict
from os.path import join
from os.path import join, expanduser
import torch
import glob
import random
import cv2
from multi_task_il_gnn.datasets import split_files
from copy import deepcopy, copy
from torchvision import transforms
from torchvision.transforms import RandomAffine, ToTensor, Normalize, \
    RandomGrayscale, ColorJitter, RandomApply, RandomHorizontalFlip, GaussianBlur, RandomResizedCrop
from torchvision.transforms.functional import resized_crop
import albumentations as A

from colorama import init as colorama_init
from colorama import Fore
from colorama import Style

colorama_init()

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
                    'single_bin': [0.15, 0.06, 0.15]}
    }
}

NUM_OBJ_NUM_TARGET_PER_OBJ = {'pick_place': (4, 4),
                              'nut_assembly': (3, 3)}

ENV_INFORMATION = {
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
    'img_dim': [200, 360]
}

NUM_FEATURES = 12

def plot_graph(data, save_path=None):
    # Extracting node features and edge indices
    x = data.x.numpy()
    edge_index = data.edge_index.numpy()

    # Creating a directed graph
    G = nx.DiGraph()

    # Adding nodes with features
    for i in range(x.shape[0]):
        G.add_node(i, feature=x[i])

    # Adding edges
    for src, dst in edge_index.T:
        G.add_edge(src, dst)

    # Getting the last feature values
    last_features = x[:, -1]

    # Plotting the graph
    pos = nx.spring_layout(G)  # positions for all nodes
    node_colors = ['blue' if feature ==
                   1 else 'red' for feature in last_features]
    nx.draw(G, pos, node_color=node_colors, node_size=300, with_labels=True, labels={i: str(i) for i in G.nodes()},
            linewidths=0.5, font_size=8)

    # Save the plot if save_path is provided
    plt.savefig("graph_debug.png")


def compute_object_features(task_name: str, object_name: str):
    if task_name == "pick_place":
        if object_name == "greenbox":
            return np.array([29, 122, 41])
        elif object_name == "yellowbox":
            return np.array([255, 250, 160])
        elif object_name == "bluebox":
            return np.array([9, 51, 93])
        elif object_name == "redbox":
            return np.array([229, 0, 20])
        elif object_name == 'bin':
            return np.array([168, 116, 76])


def create_train_val_dict(dataset_loader=object, agent_name: str = "ur5e", demo_name: str = "panda", agent_root_dir: str = "", demo_root_dir: str = "", task_spec=None, split: list = [0.9, 0.1]):

    count = 0
    agent_file_cnt = 0
    demo_file_cnt = 0
    for spec in task_spec:
        name, date = spec.get('name', None), spec.get('date', None)
        assert name, 'need to specify the task name for data generated, for easier tracking'
        dataset_loader.agent_files[name] = dict()
        dataset_loader.demo_files[name] = dict()

        if dataset_loader.mode == 'train':
            print(
                "Loading task [{:<9}] saved on date {}".format(name, date))
        if date is None:
            agent_dir = join(
                agent_root_dir, name, '{}_{}'.format(agent_name, name))

            print(f"{Fore.GREEN}Taking agent files from {agent_dir}{Style.RESET_ALL}")
            demo_dir = join(
                demo_root_dir, name, '{}_{}'.format(demo_name, name))
            print(f"{Fore.GREEN}Taking demo files from {demo_dir}{Style.RESET_ALL}")
        else:
            agent_dir = join(
                agent_root_dir, name, '{}_{}_{}'.format(date, agent_name, name))
            demo_dir = join(
                demo_root_dir, name, '{}_{}_{}'.format(date, demo_name, name))
        dataset_loader.subtask_to_idx[name] = defaultdict(list)
        dataset_loader.demo_subtask_to_idx[name] = defaultdict(list)
        for _id in range(spec.get('n_tasks')):
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

            dataset_loader.agent_files[name][_id] = deepcopy(agent_files)
            dataset_loader.demo_files[name][_id] = deepcopy(demo_files)

            for demo in demo_files:
                for agent in agent_files:
                    dataset_loader.all_file_pairs[count] = (
                        name, _id, demo, agent)
                    dataset_loader.task_to_idx[name].append(count)
                    dataset_loader.subtask_to_idx[name][task_id].append(
                        count)
                    count += 1

        print(
            f'{Fore.RED}Done loading Task {name}, agent/demo trajctores pairs reach a count of: {count}{Style.RESET_ALL}')

        if spec.get('demo_crop', None) is not None:
            dataset_loader.demo_crop[name] = spec.get(
                'demo_crop', [0, 0, 0, 0])
        if spec.get('agent_crop', None) is not None:
            dataset_loader.agent_crop[name] = spec.get(
                'agent_crop', [0, 0, 0, 0])
        if spec.get('crop', None) is not None:
            dataset_loader.task_crops[name] = spec.get(
                'crop', [0, 0, 0, 0])

    return count


def create_data_aug(dataset_loader=object):

    assert dataset_loader.data_augs, 'Must give some basic data-aug parameters'
    if dataset_loader.mode == 'train':
        print('Data aug parameters:', dataset_loader.data_augs)

    dataset_loader.toTensor = ToTensor()
    old_aug = dataset_loader.data_augs.get('old_aug', True)
    dataset_loader.transforms = transforms.Compose([
        transforms.ColorJitter(
            brightness=list(dataset_loader.data_augs.get(
                "brightness", [0.875, 1.125])),
            contrast=list(dataset_loader.data_augs.get(
                "contrast", [0.5, 1.5])),
            saturation=list(dataset_loader.data_augs.get(
                "contrast", [0.5, 1.5])),
            hue=list(dataset_loader.data_augs.get("hue", [-0.05, 0.05])),
        )
    ])

    def frame_aug(task_name, obs, second=False, bb=None, class_frame=None, perform_aug=True, frame_number=-1, perform_scale_resize=True, agent=False):

        if perform_scale_resize:
            img_height, img_width = obs.shape[:2]
            """applies to every timestep's RGB obs['camera_front_image']"""
            if len(getattr(dataset_loader, "demo_crop", OrderedDict())) != 0 and not agent:
                crop_params = dataset_loader.demo_crop.get(
                    task_name, [0, 0, 0, 0])
            if len(getattr(dataset_loader, "agent_crop", OrderedDict())) != 0 and agent:
                crop_params = dataset_loader.agent_crop.get(
                    task_name, [0, 0, 0, 0])
            if len(getattr(dataset_loader, "task_crops", OrderedDict())) != 0:
                crop_params = dataset_loader.task_crops.get(
                    task_name, [0, 0, 0, 0])

            top, left = crop_params[0], crop_params[2]
            img_height, img_width = obs.shape[0], obs.shape[1]
            box_h, box_w = img_height - top - \
                crop_params[1], img_width - left - crop_params[3]

            obs = dataset_loader.toTensor(obs)
            # ---- Resized crop ----#
            obs = resized_crop(obs, top=top, left=left, height=box_h,
                               width=box_w, size=(dataset_loader.height, dataset_loader.width))
            if DEBUG:
                cv2.imwrite(f"prova_resized_{frame_number}.png", np.moveaxis(
                    obs.numpy()*255, 0, -1))

        else:
            obs = dataset_loader.toTensor(obs)
            if bb is not None and class_frame is not None:
                for obj_indx, obj_bb in enumerate(bb):
                    # Convert normalized bounding box coordinates to actual coordinates
                    x1, y1, x2, y2 = obj_bb
                    # replace with new bb
                    bb[obj_indx] = np.array([[x1, y1, x2, y2]])

        # ---- Augmentation ----#
        if perform_aug:
            augmented = dataset_loader.transforms(obs)
        else:
            augmented = obs
        if DEBUG:
            if agent:
                cv2.imwrite("weak_augmented.png", np.moveaxis(
                    augmented.numpy()*255, 0, -1))
        assert augmented.shape == obs.shape

        return augmented
    return frame_aug


def make_demo(dataset, traj, task_name):
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
            # convert from BGR to RGB and scale to 0-1 range
            obs = copy(
                traj.get(n)['obs']['camera_front_image'][:, :, ::-1])
            processed = dataset.frame_aug(
                task_name,
                obs,
                perform_aug=False,
                frame_number=i,
                perform_scale_resize=True)
            frames.append(processed)
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
            obs = copy(
                traj.get(n)['obs']['camera_front_image'][:, :, ::-1])

            processed = dataset.frame_aug(task_name,
                                          obs,
                                          perform_aug=False,
                                          perform_scale_resize=True,
                                          agent=False)
            frames.append(processed)

    ret_dict = dict()
    ret_dict['demo'] = torch.stack(frames)
    return ret_dict
