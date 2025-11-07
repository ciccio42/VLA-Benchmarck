"""\
This file loads the trajectories in pkl format from the specified folder and add the bounding-box related to the objects in the scene
The bounding box is defined as follow: (center_x, center_y, width, height)
"""

from multi_task_il.datasets.savers import _compress_obs
import os
import sys
import pickle
import cv2
import numpy as np
import logging
import copy
import robosuite.utils.transform_utils as T
import functools
from multiprocessing import Pool, cpu_count
import glob

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger("BB-Creator")

ENV_OBJECTS = {
    'pick_place': {
        'obj_names': ['greenbox', 'yellowbox', 'bluebox', 'redbox', 'bin'],
        'bin_position': [0.18, 0.00, 0.75],
        'obj_dim': {'greenbox': [0.05, 0.055, 0.045],  # W, H, D
                    'yellowbox': [0.05, 0.055, 0.045],
                    'bluebox': [0.05, 0.055, 0.045],
                    'redbox': [0.05, 0.055, 0.045],
                    'bin': [0.6, 0.06, 0.15]},
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

OFFSET = 0.0  # CM


def plot_bb(img, obj_bb):

    # draw bb
    for obj_name in obj_bb.keys():
        center = obj_bb[obj_name]['center']
        upper_left_corner = obj_bb[obj_name]['upper_left_corner']
        bottom_right_corner = obj_bb[obj_name]['bottom_right_corner']
        img = cv2.circle(
            img, center, radius=1, color=(0, 0, 255), thickness=-1)
        img = cv2.rectangle(
            img, upper_left_corner,
            bottom_right_corner, (255, 0, 0), 1)
    cv2.imwrite("test_bb.png", img)
    # cv2.imshow("Test", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def overwrite_pkl_file(pkl_file_path, sample, traj_obj_bb):
    # get trajectory from sample
    new_sample = copy.deepcopy(sample)

    traj = new_sample['traj']

    # modify trajectory observation
    for t in range(len(traj)):
        try:
            obs = traj.get(t)['obs']
        except:
            _img = traj._data[t][0]['camera_front_image']
            okay, im_string = cv2.imencode(
                '.jpg', _img)
            traj._data[t][0]['camera_front_image'] = im_string
            obs = traj.get(t)['obs']

        obs['obj_bb'] = traj_obj_bb[t]
        obs = _compress_obs(obs)
        traj.change_obs(t, obs)
        logger.debug(obs.keys())

    pickle.dump({
        'traj': traj,
        'len': len(traj),
        'env_type': sample['env_type'],
        'task_id': sample['task_id']}, open(pkl_file_path, 'wb'))


def write_bb(pkl_file_path):
    # pkl_file_path = os.path.join(task_path, pkl_file_path)
    # logger.info(f"Task id {dir} - Trajectory {pkl_file_path}")
    # 2. Load pickle file
    with open(pkl_file_path, "rb") as f:
        sample = pickle.load(f)
    # 3. Identify objects positions
    object_name_list = ENV_OBJECTS[args.task_name]['obj_names']
    traj = sample['traj']
    traj_bb = []
    for t in range(len(sample['traj'])):
        # for each object in the observation get the position
        obj_positions = dict()
        obj_bb = dict()
        if args.task_name == 'pick_place':
            try:
                obs = traj.get(t)['obs']
                print(traj.keys())
            except:
                _img = traj._data[t][0]['camera_front_image']
                okay, im_string = cv2.imencode(
                    '.jpg', _img)
                traj._data[t][0]['camera_front_image'] = im_string
                obs = traj.get(t)['obs']

            logger.debug(obs.keys())
            for obj_name in object_name_list:
                if obj_name != 'bin':
                    obj_positions[obj_name] = obs[f"{obj_name}_pos"]
                else:
                    obj_positions[obj_name] = ENV_OBJECTS[args.task_name]['bin_position']

        for camera_name in ENV_OBJECTS[args.task_name]['camera_names']:
            # 1. Compute rotation_camera_to_world
            camera_quat = ENV_OBJECTS[args.task_name]['camera_orientation'][camera_name]
            r_camera_world = T.quat2mat(
                T.convert_quat(np.array(camera_quat), to='xyzw')).T
            p_camera_world = - \
                r_camera_world @ np.array(
                    ENV_OBJECTS[args.task_name]['camera_pos'][camera_name]).T

            obj_bb[camera_name] = dict()

            # for each object create bb
            for obj_name in object_name_list:
                obj_bb[camera_name][obj_name] = dict()

                logger.debug(f"\nObject: {obj_name}")
                # convert obj pos in camera coordinate
                obj_pos = obj_positions[obj_name]
                obj_pos = np.array([obj_pos])

                # 2. Create transformation matrix
                T_camera_world = np.concatenate(
                    (r_camera_world, p_camera_world), axis=1)
                T_camera_world = np.concatenate(
                    (T_camera_world, np.array([[0, 0, 0, 1]])), axis=0)
                # logger.debug(T_camera_world)
                p_world_object = np.expand_dims(
                    np.insert(obj_pos, 3, 1), 0).T
                p_camera_object = T_camera_world @ p_world_object
                logger.debug(
                    f"\nP_world_object:\n{p_world_object} - \nP_camera_object:\n {p_camera_object}")

                # 3. Cnversion into pixel coordinates of object center
                f = 0.5 * ENV_OBJECTS[args.task_name]['img_dim'][0] / \
                    np.tan(ENV_OBJECTS[args.task_name]
                           ['camera_fovy'] * np.pi / 360)

                p_x_center = int(
                    (p_camera_object[0][0] / - p_camera_object[2][0]) * f + ENV_OBJECTS[args.task_name]['img_dim'][1] / 2)

                p_y_center = int(
                    (- p_camera_object[1][0] / - p_camera_object[2][0]) * f + ENV_OBJECTS[args.task_name]['img_dim'][0] / 2)
                logger.debug(
                    f"\nImage coordinate: px {p_x_center}, py {p_y_center}")

                p_x_corner_list = []
                p_y_corner_list = []
                # 3.1 create a box around the object
                for i in range(8):
                    if i == 0:  # upper-left front corner
                        p_world_object_corner = p_world_object + \
                            np.array(
                                [[ENV_OBJECTS[args.task_name]
                                    ['obj_dim'][obj_name][2]/2],
                                    [-ENV_OBJECTS[args.task_name]
                                     ['obj_dim'][obj_name][0]/2-OFFSET],
                                    [ENV_OBJECTS[args.task_name]
                                     ['obj_dim'][obj_name][1]/2+OFFSET],
                                    [0]])
                    elif i == 1:  # upper-right front corner
                        p_world_object_corner = p_world_object + \
                            np.array(
                                [[ENV_OBJECTS[args.task_name]
                                    ['obj_dim'][obj_name][2]/2],
                                    [ENV_OBJECTS[args.task_name]
                                     ['obj_dim'][obj_name][0]/2+OFFSET],
                                    [ENV_OBJECTS[args.task_name]
                                     ['obj_dim'][obj_name][1]/2+OFFSET],
                                    [0]])
                    elif i == 2:  # bottom-left front corner
                        p_world_object_corner = p_world_object + \
                            np.array(
                                [[ENV_OBJECTS[args.task_name]
                                    ['obj_dim'][obj_name][2]/2],
                                    [-ENV_OBJECTS[args.task_name]
                                     ['obj_dim'][obj_name][0]/2-OFFSET],
                                    [-ENV_OBJECTS[args.task_name]
                                     ['obj_dim'][obj_name][1]/2-OFFSET],
                                    [0]])
                    elif i == 3:  # bottom-right front corner
                        p_world_object_corner = p_world_object + \
                            np.array(
                                [[ENV_OBJECTS[args.task_name]
                                    ['obj_dim'][obj_name][2]/2],
                                    [ENV_OBJECTS[args.task_name]
                                     ['obj_dim'][obj_name][0]/2+OFFSET],
                                    [-ENV_OBJECTS[args.task_name]
                                     ['obj_dim'][obj_name][1]/2-OFFSET],
                                    [0]])
                    elif i == 4:  # upper-left back corner
                        p_world_object_corner = p_world_object + \
                            np.array(
                                [[-ENV_OBJECTS[args.task_name]
                                    ['obj_dim'][obj_name][2]/2],
                                    [-ENV_OBJECTS[args.task_name]
                                     ['obj_dim'][obj_name][0]/2-OFFSET],
                                    [ENV_OBJECTS[args.task_name]
                                     ['obj_dim'][obj_name][1]/2+OFFSET],
                                    [0]])
                    elif i == 5:  # upper-right back corner
                        p_world_object_corner = p_world_object + \
                            np.array(
                                [[-ENV_OBJECTS[args.task_name]
                                    ['obj_dim'][obj_name][2]/2],
                                    [ENV_OBJECTS[args.task_name]
                                     ['obj_dim'][obj_name][0]/2+OFFSET],
                                    [ENV_OBJECTS[args.task_name]
                                     ['obj_dim'][obj_name][1]/2+OFFSET],
                                    [0]])
                    elif i == 6:  # bottom-left back corner
                        p_world_object_corner = p_world_object + \
                            np.array(
                                [[-ENV_OBJECTS[args.task_name]
                                    ['obj_dim'][obj_name][2]/2],
                                    [-ENV_OBJECTS[args.task_name]
                                     ['obj_dim'][obj_name][0]/2-OFFSET],
                                    [-ENV_OBJECTS[args.task_name]
                                     ['obj_dim'][obj_name][1]/2-OFFSET],
                                    [0]])
                    elif i == 7:  # bottom-right back corner
                        p_world_object_corner = p_world_object + \
                            np.array(
                                [[-ENV_OBJECTS[args.task_name]
                                    ['obj_dim'][obj_name][2]/2],
                                    [ENV_OBJECTS[args.task_name]
                                     ['obj_dim'][obj_name][0]/2+OFFSET],
                                    [-ENV_OBJECTS[args.task_name]
                                     ['obj_dim'][obj_name][1]/2-OFFSET],
                                    [0]])

                    p_camera_object_corner = T_camera_world @ p_world_object_corner
                    logger.debug(
                        f"\nP_world_object_upper_left:\n{p_world_object_corner} -   \nP_camera_object_upper_left:\n {p_camera_object_corner}")

                    # 3.1 Upper-left corner and bottom right corner in pixel coordinate
                    p_x_corner = int(
                        (p_camera_object_corner[0][0] / - p_camera_object_corner[2][0]) * f + ENV_OBJECTS[args.task_name]['img_dim'][1] / 2)

                    p_y_corner = int(
                        (- p_camera_object_corner[1][0] / - p_camera_object_corner[2][0]) * f + ENV_OBJECTS[args.task_name]['img_dim'][0] / 2)
                    logger.debug(
                        f"\nImage coordinate upper_left corner: px {p_x_corner}, py {p_y_corner}")

                    p_x_corner_list.append(p_x_corner)
                    p_y_corner_list.append(p_y_corner)

                x_min = min(p_x_corner_list)
                y_min = min(p_y_corner_list)
                x_max = max(p_x_corner_list)
                y_max = max(p_y_corner_list)
                # save bb
                obj_bb[camera_name][obj_name]['center'] = [
                    p_x_center, p_y_center]
                obj_bb[camera_name][obj_name]['upper_left_corner'] = [
                    x_max, y_max]
                obj_bb[camera_name][obj_name]['bottom_right_corner'] = [
                    x_min, y_min]
                if obj_name == 'bin':
                    print(obj_bb)
            # draw center
            # if camera_name == 'camera_front':
            #     img = np.array(traj.get(
            #         t)['obs'][f'{camera_name}_image'][:, :, ::-1])
            #     plot_bb(img, obj_bb[camera_name])

        # print(obj_bb)
        traj_bb.append(obj_bb)

    # save sample with objects bb
    overwrite_pkl_file(pkl_file_path=pkl_file_path,
                       sample=sample,
                       traj_obj_bb=traj_bb)


if __name__ == '__main__':
    import debugpy
    debugpy.listen(('0.0.0.0', 5678))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_path', default="/", help="Path to task")
    parser.add_argument('--task_name', default="/", help="Name of the task")
    parser.add_argument('--robot_name', default="/", help="Name of the robot")
    args = parser.parse_args()

    # 1. Load the dataset
    folder_path = os.path.join(
        args.task_path, args.task_name, f"{args.robot_name}_{args.task_name}")

    for dir in os.listdir(folder_path):
        print(dir)
        if "task_" in dir:
            task_path = os.path.join(folder_path, dir)
            print(task_path)
            i = 0
            trj_list = glob.glob(f"{task_path}/*.pkl")
            with Pool(1) as p:
                f = functools.partial(write_bb)
                p.map(f, trj_list)
