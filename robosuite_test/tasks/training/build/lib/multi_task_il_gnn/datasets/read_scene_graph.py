import pickle as pkl
import glob
import os
import numpy as np
from robosuite.utils.transform_utils import quat2axisangle
from torch_geometric.data import Data
import torch

"""_summary_
pick-place task mapping:
0: green block
1: yellow block
2: blue block
3: red block
4: first bin (the bin at the left)
5: second bin
6: third bin
7: fourth bin (the bin at the right)
# Type: Object 1, Placing 0
# Target_task: Target 1, No-Target 0
vector = [p_x,p_y, p_z, r_x, r_y, r_z, obj_class, type, target_task]
"""

NUM_FEATURES = 12


def read_pkl(file_path: str):
    with open(file_path, 'rb') as f:
        traj = pkl.load(f)
    return traj


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_folder_path', default=None, type=str)
    parser.add_argument('--task_name', default="pick_place", type=str)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    task_folder_paths = glob.glob(os.path.join(
        args.task_folder_path, "task_*"))

    robot_name = args.task_folder_path.split('/')[-1].split('_')[0]
    print(f"Considering robot {robot_name}")
    for task_path in task_folder_paths:
        task_variation = task_path.split('/')[-1]
        print(f"Considering task {task_path.split('/')[-1]}")
        pkl_files_path = glob.glob(os.path.join(
            task_path, "*.pkl"))

        for pkl_file_path in pkl_files_path:
            pkl_file_name = pkl_file_path.split('/')[-1].split('.')[0]
            scene_graphs = read_pkl(file_path=pkl_file_path)
            plot_graph(scene_graphs[0])
