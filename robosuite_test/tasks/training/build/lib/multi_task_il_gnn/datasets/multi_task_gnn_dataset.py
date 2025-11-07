import torch
from torch_geometric.data import Dataset, download_url
from collections import OrderedDict, defaultdict
from multi_task_il_gnn.datasets.utils import create_train_val_dict, create_data_aug, make_demo, NUM_OBJ_NUM_TARGET_PER_OBJ
import time
from multi_task_il_gnn.datasets import load_traj, load_graph
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')

# Create a logger object
logger = logging.getLogger('GNN-Dataset')


class MultiTaskGNNDataset(Dataset):
    def __init__(
            self,
            tasks_spec,
            agent_root_dir='mosaic_multitask_dataset',
            demo_root_dir='mosaic_multitask_dataset',
            mode='train',
            split=[0.9, 0.1],
            demo_T=4,
            width=180,
            height=100,
            data_augs=None,
            non_sequential=False,
            agent_name='ur5e',
            demo_name='panda',
            select_random_frames=True
    ):

        self.task_crops = OrderedDict()
        self.demo_crop = OrderedDict()
        self.agent_crop = OrderedDict()
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
        self.select_random_frames = select_random_frames

        # Frame distribution for each trajectory
        self._frame_distribution = OrderedDict()
        self._mix_demo_agent = False
        count = create_train_val_dict(self,
                                      agent_name,
                                      demo_name,
                                      agent_root_dir,
                                      demo_root_dir,
                                      tasks_spec,
                                      split)
        self.pairs_count = count

        self.task_count = len(tasks_spec)

        self._demo_T = demo_T
        self.width, self.height = width, height

        self.non_sequential = non_sequential
        if non_sequential:
            print("Warning! The agent observations are not sampled in neighboring timesteps, make sure inverse dynamics loss is NOT used in training \n ")

        self.data_augs = data_augs
        self.frame_aug = create_data_aug(self)

    def __len__(self):
        return self.pairs_count

    def __getitem__(self, idx):
        (task_name, sub_task_id, demo_file,
         agent_file) = self.all_file_pairs[idx]

        start = time.time()
        demo_traj, agent_graph = load_traj(demo_file), load_graph(agent_file)
        end = time.time()
        logger.debug(f"Loading time {end-start}")

        # take node features from graph
        node_features = agent_graph.x
        # take node class from graph
        class_labels = agent_graph.y
        if task_name == 'pick_place':
            num_objs = NUM_OBJ_NUM_TARGET_PER_OBJ[task_name][0]
            num_targets = NUM_OBJ_NUM_TARGET_PER_OBJ[task_name][1]

            obj_class = torch.zeros(num_objs+num_targets)
            target_class = torch.zeros(num_objs+num_targets)

            obj_class[:num_objs] = class_labels[:num_objs]
            target_class[num_objs:] = class_labels[num_objs:]

        # start = time.time()
        demo_data = make_demo(self, demo_traj[0], task_name)

        return {'demo_data': demo_data, 'node_features': node_features, 'obj_class': obj_class, 'target_class': target_class, 'task_name': task_name, 'task_id': sub_task_id}

    def _make_traj(self, traj, command, task_name, sub_task_id):
        pass
