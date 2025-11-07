import torch
from torch_geometric.data import Data
from robosuite.utils.transform_utils import quat2axisangle
import numpy as np
import pickle as pkl
import glob
import os
from multi_task_il_gnn.datasets.utils import OBJECTS_POS_DIM, NUM_OBJ_NUM_TARGET_PER_OBJ, NUM_FEATURES, compute_object_features, plot_graph

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


def read_pkl(file_path: str):
    with open(file_path, 'rb') as f:
        traj = pkl.load(f)
    return traj


def write_pkl(file_path: str, obj: object):
    with open(file_path, 'wb') as f:
        pkl.dump(obj, f)


def generate_scene_graph(traj, task_name):
    graphs = list()

    for t in range(len(traj)):
        # get the current observation
        obs = traj[t].get('obs')

        if task_name == "pick_place":
            num_entities = 8

        # 1. Fill the feature_vector matrix
        # position, orientation, RGB, object_class, type, target_task # 12
        feature_vector = np.zeros((num_entities, NUM_FEATURES))
        for obj_indx, object_name in enumerate(OBJECTS_POS_DIM[task_name]['obj_names']):
            if 'bin' != object_name:
                object_position = obs[f'{object_name}_pos']
                object_orientation = quat2axisangle(obs[f'{object_name}_quat'])
            else:
                object_position = OBJECTS_POS_DIM[task_name]['bin_position']

            if task_name == "pick_place":
                if "bin" not in object_name:
                    feature_vector[obj_indx][:3] = object_position
                    feature_vector[obj_indx][3:6] = object_orientation
                    feature_vector[obj_indx][6:9] = compute_object_features(task_name=task_name,
                                                                            object_name=object_name)
                    feature_vector[obj_indx][9] = np.array(
                        obj_indx, dtype=np.uint8)
                    feature_vector[obj_indx][10] = np.array(1, dtype=np.uint8)
                    feature_vector[obj_indx][11] = 1 if obj_indx == obs['target-object'] else 0

                else:
                    # we are considering bins
                    bin_pos = OBJECTS_POS_DIM[task_name]['bin_position']
                    bins_pos = list()
                    bins_pos.append([bin_pos[0],
                                     bin_pos[1]-0.15-0.15/2,
                                     bin_pos[2]])
                    bins_pos.append([bin_pos[0],
                                    bin_pos[1]-0.15/2,
                                    bin_pos[2]])
                    bins_pos.append([bin_pos[0],
                                    bin_pos[1]+0.15/2,
                                    bin_pos[2]])
                    bins_pos.append([bin_pos[0],
                                    bin_pos[1]+0.15+0.15/2,
                                    bin_pos[2]])
                    for bin_indx, pos in enumerate(bins_pos):
                        bin_indx_relative = bin_indx + \
                            (len(OBJECTS_POS_DIM[task_name]['obj_names'])-1)
                        feature_vector[bin_indx_relative][:3] = pos
                        feature_vector[bin_indx_relative][3:
                                                          6] = np.array([.0, .0, .0])
                        feature_vector[bin_indx_relative][6:9] = compute_object_features(task_name=task_name,
                                                                                         object_name=object_name)
                        feature_vector[bin_indx_relative][9] = np.array(
                            bin_indx+bin_indx_relative, dtype=np.uint8)
                        feature_vector[bin_indx_relative][10] = np.array(
                            0, dtype=np.uint8)
                        feature_vector[bin_indx_relative][11] = 1 if bin_indx == obs['target-box-id'] else 0

        # 2. Fill the edge index vector
        # connect each object with a target position
        # edge_indx has shape 2*num_edge
        num_edges = (NUM_OBJ_NUM_TARGET_PER_OBJ[task_name][0] *
                     NUM_OBJ_NUM_TARGET_PER_OBJ[task_name][1]) * 2
        edge_indx = np.zeros((2, num_edges))

        num_edge = 0
        for object_indx in range(NUM_OBJ_NUM_TARGET_PER_OBJ[task_name][0]):
            for place_indx in range(NUM_OBJ_NUM_TARGET_PER_OBJ[task_name][1]):
                # undirected graph edges
                edge_indx[0][num_edge] = object_indx
                edge_indx[1][num_edge] = place_indx + \
                    NUM_OBJ_NUM_TARGET_PER_OBJ[task_name][0]  # offset in node indices
                num_edge += 1

                edge_indx[0][num_edge] = place_indx + \
                    NUM_OBJ_NUM_TARGET_PER_OBJ[task_name][0]
                edge_indx[1][num_edge] = object_indx
                num_edge += 1

        graph = Data(x=torch.tensor(feature_vector[:, :-1]),
                     edge_index=torch.tensor(edge_indx),
                     y=torch.tensor(feature_vector[:, -1]))
        # plot_graph(data=graph)
        graphs.append(graph)

    return graphs


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_folder_path', default=None, type=str)
    parser.add_argument('--task_name', default="pick_place", type=str)
    parser.add_argument(
        '--save_path',
        default="/user/frosa/multi_task_lfd/ur_multitask_dataset/geometric_graphs",
        type=str)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    task_folder_paths = glob.glob(os.path.join(
        args.task_folder_path, "task_*"))

    save_folder = os.path.join(args.save_path, args.task_name)
    os.makedirs(save_folder,
                exist_ok=True)

    robot_name = args.task_folder_path.split('/')[-1].split('_')[0]
    print(f"Considering robot {robot_name}")
    for task_path in task_folder_paths:
        task_variation = task_path.split('/')[-1]
        print(f"Considering task {task_path.split('/')[-1]}")
        pkl_files_path = glob.glob(os.path.join(
            task_path, "*.pkl"))

        for pkl_file_path in pkl_files_path:
            pkl_file_name = pkl_file_path.split('/')[-1].split('.')[0]
            traj = read_pkl(file_path=pkl_file_path)['traj']

            scene_graph = generate_scene_graph(
                traj=traj, task_name=args.task_name)

            # save scene_graph
            save_file_path = os.path.join(
                save_folder, f"{robot_name}_{args.task_name}", task_variation)
            os.makedirs(save_file_path, exist_ok=True)
            save_file_path = os.path.join(
                save_file_path, f"{pkl_file_name}.pkl")
            # write_pkl(file_path=save_file_path,
            #           obj=scene_graph)
