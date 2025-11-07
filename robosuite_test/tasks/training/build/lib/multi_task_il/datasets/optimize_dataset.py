"""\
This file loads the trajectories in pkl format from the specified folder and add the bounding-box related to the objects in the scene
The bounding box is defined as follow: (center_x, center_y, width, height)
"""
import yaml
from torchvision.transforms.functional import resized_crop
from torchvision.transforms import ToTensor
import glob
from multiprocessing import Pool, cpu_count
import functools
import robosuite.utils.transform_utils as T
import copy
import logging
import numpy as np
import cv2
import pickle
import sys
from multi_task_il.datasets.savers import _compress_obs
import os
from multi_task_il.datasets.utils import OBJECTS_POS_DIM

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger("BB-Creator")

KEY_INTEREST = ["joint_pos", "joint_vel", "eef_pos",
                "eef_quat", "gripper_qpos", "gripper_qvel", "camera_front_image",
                "target-box-id", "target-object", "obj_bb",
                "extent", "zfar", "znear", "eef_point", "ee_aa", "target-peg"]
OFFSET = 0.0
WORKERS = 1


def crop_resize_img(task_cfg, task_name, obs, bb):
    """applies to every timestep's RGB obs['camera_front_image']"""
    crop_params = task_cfg[task_name].get('agent_crop', [0, 0, 0, 0])
    top, left = crop_params[0], crop_params[2]
    img_height, img_width = obs.shape[0], obs.shape[1]
    box_h, box_w = img_height - top - \
        crop_params[1], img_width - left - crop_params[3]

    cropped_img = obs[top:box_h, left:box_w]
    cv2.imwrite("cropped.jpg", cropped_img)

    img_res = cv2.resize(cropped_img, (180, 100))
    adj_bb = None
    if bb is not None:
        adj_bb = adjust_bb(bb,
                           obs,
                           cropped_img,
                           img_res,
                           img_width=img_width,
                           img_height=img_height,
                           top=top,
                           left=left,
                           box_w=box_w,
                           box_h=box_h)

    return img_res, adj_bb


def adjust_bb(bb, original, cropped_img, obs, img_width=360, img_height=200, top=0, left=0, box_w=360, box_h=200):
    # For each bounding box
    for obj_indx, obj_bb_name in enumerate(bb):
        obj_bb = np.concatenate(
            (bb[obj_bb_name]['bottom_right_corner'], bb[obj_bb_name]['upper_left_corner']))
        # Convert normalized bounding box coordinates to actual coordinates
        x1_old, y1_old, x2_old, y2_old = obj_bb
        x1_old = int(x1_old)
        y1_old = int(y1_old)
        x2_old = int(x2_old)
        y2_old = int(y2_old)

        # Modify bb based on computed resized-crop
        # 1. Take into account crop and resize
        x_scale = obs.shape[1]/cropped_img.shape[1]
        y_scale = obs.shape[0]/cropped_img.shape[0]
        x1 = int(np.round((x1_old - left) * x_scale))
        x2 = int(np.round((x2_old - left) * x_scale))
        y1 = int(np.round((y1_old - top) * y_scale))
        y2 = int(np.round((y2_old - top) * y_scale))

        # image = cv2.rectangle(original,
        #                       (x1_old,
        #                        y1_old),
        #                       (x2_old,
        #                        y2_old),
        #                       color=(0, 0, 255),
        #                       thickness=1)
        # cv2.imwrite("bb_original.png", image)

        # image = cv2.rectangle(cropped_img,
        #                       (int((x1_old - left)),
        #                        int((y1_old - top))),
        #                       (int((x2_old - left)),
        #                        int((y2_old - top))),
        #                       color=(0, 0, 255),
        #                       thickness=1)
        # cv2.imwrite("bb_cropped.png", image)

        # image = cv2.rectangle(obs,
        #                       (x1,
        #                        y1),
        #                       (x2,
        #                        y2),
        #                       color=(0, 0, 255),
        #                       thickness=1)
        # cv2.imwrite("bb_cropped_resize.png", image)

        # replace with new bb
        bb[obj_bb_name]['bottom_right_corner'] = np.array([x2, y2])
        bb[obj_bb_name]['upper_left_corner'] = np.array([x1, y1])
        bb[obj_bb_name]['center'] = np.array([int((x2-x1)/2), int((y2-y1)/2)])
    return bb


def scale_bb(center, upper_left, bottom_right, reduction=0.5):
    width = bottom_right[0] - upper_left[0]
    height = bottom_right[1] - upper_left[1]

    new_width = int(width * reduction)
    new_height = int(height * reduction)

    new_upper_left = (center[0] - new_width // 2, center[1] - new_height // 2)
    new_bottom_right = (center[0] + new_width // 2,
                        center[1] + new_height // 2)

    return new_upper_left, new_bottom_right


def compute_displacemet(first_bb, last_bb):
    displacement_center = abs(
        np.array(first_bb['center']) - np.array(last_bb['center']))
    width = abs(first_bb['upper_left_corner'][0] -
                first_bb['bottom_right_corner'][0])
    height = abs(first_bb['upper_left_corner'][0] -
                 first_bb['bottom_right_corner'][0])
    return displacement_center, width, height


def compute_new_bb(displacement, width, height, bb, obj_name):
    old_center = bb['center']
    old_upper_left = bb['upper_left_corner']
    old_bottom_right = bb['bottom_right_corner']

    if "machine1" in obj_name:
        new_center = [old_center[0] + displacement[0],
                      old_center[1] + displacement[1]]
        new_upper_left = [old_upper_left[0] + displacement[0],
                          old_upper_left[1] + displacement[1]]
        new_bottom_right = [old_bottom_right[0] +
                            displacement[0], old_bottom_right[1] + displacement[1]]
    elif "machine2" in obj_name:
        new_center = [old_center[0] - displacement[0],
                      old_center[1] - displacement[1]]
        new_upper_left = [old_upper_left[0] - displacement[0],
                          old_upper_left[1] - displacement[1]]
        new_bottom_right = [old_bottom_right[0] -
                            displacement[0], old_bottom_right[1] - displacement[1]]

    return {
        'center': new_center,
        'upper_left_corner': new_upper_left,
        'bottom_right_corner': new_bottom_right
    }


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


def opt_traj(task_name, task_spec, out_path, rescale_bb, real, pkl_file_path):
    # pkl_file_path = os.path.join(task_path, pkl_file_path)
    # logger.info(f"Task id {dir} - Trajectory {pkl_file_path}")
    # 2. Load pickle file
    with open(pkl_file_path, "rb") as f:
        sample = pickle.load(f)

    keys = list(sample['traj']._data[0][0].keys())
    keys_to_remove = []
    for key in keys:
        if key not in KEY_INTEREST:
            keys_to_remove.append(key)

    # remove data not of interest for training
    start_pick_t = 0
    # end_pick_t = 0
    if 'press_button' in task_name:
        # I have to get the last frame to identify the final placing position
        last_step_obs = sample['traj'][len(sample['traj'])-1]['obs']
        last_bb_for_all_cameras = copy.deepcopy(last_step_obs['obj_bb'])

    for t in range(len(sample['traj'])):
        for key in keys_to_remove:
            try:
                sample['traj']._data[t][0].pop(key)
            except:
                pass
        if t != 0:
            if "pick_place" in task_name and not real:
                obj_name = "single_bin"
                obj_bb = dict()
                bin_pos = OBJECTS_POS_DIM[task_name]['bin_position']

                # get bins positions
                bins_pos = []
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
                for camera_name in ["camera_front"]:
                    for bin_indx, bin_pos in enumerate(bins_pos):
                        sample['traj']._data[t][0]['obj_bb'][camera_name][f"{obj_name}_{bin_indx}"] = dict(
                        )
                        # 1. Compute rotation_camera_to_world
                        camera_quat = OBJECTS_POS_DIM[args.task_name]['camera_orientation'][camera_name]
                        r_camera_world = T.quat2mat(
                            T.convert_quat(np.array(camera_quat), to='xyzw')).T
                        p_camera_world = - \
                            r_camera_world @ np.array(
                                OBJECTS_POS_DIM[args.task_name]['camera_pos'][camera_name]).T

                        logger.debug(f"\nObject: bin")
                        # convert obj pos in camera coordinate
                        obj_pos = bin_pos
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
                        f = 0.5 * OBJECTS_POS_DIM[args.task_name]['img_dim'][0] / \
                            np.tan(OBJECTS_POS_DIM[args.task_name]
                                   ['camera_fovy'] * np.pi / 360)

                        p_x_center = int(
                            (p_camera_object[0][0] / - p_camera_object[2][0]) * f + OBJECTS_POS_DIM[args.task_name]['img_dim'][1] / 2)

                        p_y_center = int(
                            (- p_camera_object[1][0] / - p_camera_object[2][0]) * f + OBJECTS_POS_DIM[args.task_name]['img_dim'][0] / 2)
                        logger.debug(
                            f"\nImage coordinate: px {p_x_center}, py {p_y_center}")

                        image = cv2.circle(sample['traj'].get(
                            t)['obs'][f"{camera_name}_image"].copy(),
                            (p_x_center, p_y_center),
                            color=(255, 0, 0),
                            thickness=2,
                            radius=1)
                        cv2.imwrite("prova_bin_points.jpg", image)
                        p_x_corner_list = []
                        p_y_corner_list = []
                        # 3.1 create a box around the object
                        for i in range(8):
                            if i == 0:  # upper-left front corner
                                p_world_object_corner = p_world_object + \
                                    np.array(
                                        [[OBJECTS_POS_DIM[args.task_name]
                                            ['obj_dim'][obj_name][2]/2],
                                            [-OBJECTS_POS_DIM[args.task_name]
                                                ['obj_dim'][obj_name][0]/2-OFFSET],
                                            [OBJECTS_POS_DIM[args.task_name]
                                                ['obj_dim'][obj_name][1]/2+OFFSET],
                                            [0]])
                            elif i == 1:  # upper-right front corner
                                p_world_object_corner = p_world_object + \
                                    np.array(
                                        [[OBJECTS_POS_DIM[args.task_name]
                                            ['obj_dim'][obj_name][2]/2],
                                            [OBJECTS_POS_DIM[args.task_name]
                                                ['obj_dim'][obj_name][0]/2+OFFSET],
                                            [OBJECTS_POS_DIM[args.task_name]
                                                ['obj_dim'][obj_name][1]/2+OFFSET],
                                            [0]])
                            elif i == 2:  # bottom-left front corner
                                p_world_object_corner = p_world_object + \
                                    np.array(
                                        [[OBJECTS_POS_DIM[args.task_name]
                                            ['obj_dim'][obj_name][2]/2],
                                            [-OBJECTS_POS_DIM[args.task_name]
                                                ['obj_dim'][obj_name][0]/2-OFFSET],
                                            [-OBJECTS_POS_DIM[args.task_name]
                                                ['obj_dim'][obj_name][1]/2-OFFSET],
                                            [0]])
                            elif i == 3:  # bottom-right front corner
                                p_world_object_corner = p_world_object + \
                                    np.array(
                                        [[OBJECTS_POS_DIM[args.task_name]
                                            ['obj_dim'][obj_name][2]/2],
                                            [OBJECTS_POS_DIM[args.task_name]
                                                ['obj_dim'][obj_name][0]/2+OFFSET],
                                            [-OBJECTS_POS_DIM[args.task_name]
                                                ['obj_dim'][obj_name][1]/2-OFFSET],
                                            [0]])
                            elif i == 4:  # upper-left back corner
                                p_world_object_corner = p_world_object + \
                                    np.array(
                                        [[-OBJECTS_POS_DIM[args.task_name]
                                            ['obj_dim'][obj_name][2]/2],
                                            [-OBJECTS_POS_DIM[args.task_name]
                                                ['obj_dim'][obj_name][0]/2-OFFSET],
                                            [OBJECTS_POS_DIM[args.task_name]
                                                ['obj_dim'][obj_name][1]/2+OFFSET],
                                            [0]])
                            elif i == 5:  # upper-right back corner
                                p_world_object_corner = p_world_object + \
                                    np.array(
                                        [[-OBJECTS_POS_DIM[args.task_name]
                                            ['obj_dim'][obj_name][2]/2],
                                            [OBJECTS_POS_DIM[args.task_name]
                                                ['obj_dim'][obj_name][0]/2+OFFSET],
                                            [OBJECTS_POS_DIM[args.task_name]
                                                ['obj_dim'][obj_name][1]/2+OFFSET],
                                            [0]])
                            elif i == 6:  # bottom-left back corner
                                p_world_object_corner = p_world_object + \
                                    np.array(
                                        [[-OBJECTS_POS_DIM[args.task_name]
                                            ['obj_dim'][obj_name][2]/2],
                                            [-OBJECTS_POS_DIM[args.task_name]
                                                ['obj_dim'][obj_name][0]/2-OFFSET],
                                            [-OBJECTS_POS_DIM[args.task_name]
                                                ['obj_dim'][obj_name][1]/2-OFFSET],
                                            [0]])
                            elif i == 7:  # bottom-right back corner
                                p_world_object_corner = p_world_object + \
                                    np.array(
                                        [[-OBJECTS_POS_DIM[args.task_name]
                                            ['obj_dim'][obj_name][2]/2],
                                            [OBJECTS_POS_DIM[args.task_name]
                                                ['obj_dim'][obj_name][0]/2+OFFSET],
                                            [-OBJECTS_POS_DIM[args.task_name]
                                                ['obj_dim'][obj_name][1]/2-OFFSET],
                                            [0]])

                            p_camera_object_corner = T_camera_world @ p_world_object_corner
                            logger.debug(
                                f"\nP_world_object_upper_left:\n{p_world_object_corner} -   \nP_camera_object_upper_left:\n {p_camera_object_corner}")

                            # 3.1 Upper-left corner and bottom right corner in pixel coordinate
                            p_x_corner = int(
                                (p_camera_object_corner[0][0] / - p_camera_object_corner[2][0]) * f + OBJECTS_POS_DIM[args.task_name]['img_dim'][1] / 2)

                            p_y_corner = int(
                                (- p_camera_object_corner[1][0] / - p_camera_object_corner[2][0]) * f + OBJECTS_POS_DIM[args.task_name]['img_dim'][0] / 2)
                            logger.debug(
                                f"\nImage coordinate upper_left corner: px {p_x_corner}, py {p_y_corner}")

                            p_x_corner_list.append(p_x_corner)
                            p_y_corner_list.append(p_y_corner)

                        x_min = min(p_x_corner_list)
                        y_min = min(p_y_corner_list)
                        x_max = max(p_x_corner_list)
                        y_max = max(p_y_corner_list)
                        # save bb
                        sample['traj']._data[t][0]['obj_bb'][camera_name][f"{obj_name}_{bin_indx}"]['center'] = [
                            p_x_center, p_y_center]
                        sample['traj']._data[t][0]['obj_bb'][camera_name][f"{obj_name}_{bin_indx}"]['upper_left_corner'] = [
                            x_max, y_max]
                        sample['traj']._data[t][0]['obj_bb'][camera_name][f"{obj_name}_{bin_indx}"]['bottom_right_corner'] = [
                            x_min, y_min]
                        if obj_name == 'bin':
                            print(obj_bb)

                        image = cv2.rectangle(sample['traj'].get(
                            t)['obs'][f"{camera_name}_image"].copy(),
                            (x_min, y_min),
                            (x_max, y_max),
                            color=(255, 0, 0),
                            thickness=2)
                        if t == len(sample['traj'])-1:
                            cv2.imwrite("prova_bin_bb.jpg", image)
            elif 'press_button' in task_name:
                for camera_name in ["camera_front"]:
                    last_bb_all_obj = last_bb_for_all_cameras[camera_name]
                    obj_names = last_bb_all_obj.keys()

                    if t == 1:
                        # compute displacement between first and last frame for target object
                        displacement, width, height = compute_displacemet(
                            first_bb=sample['traj']._data[t][0]['obj_bb'][camera_name][sample['traj']._data[t]
                                                                                       [0]['target-object']],
                            last_bb=last_bb_all_obj[sample['traj']._data[t][0]['target-object']])

                        for obj_name in obj_names:
                            if obj_name != sample['traj']._data[t][0]['target-object']:
                                last_bb_all_obj[obj_name] = compute_new_bb(
                                    displacement,
                                    width,
                                    height,
                                    sample['traj']._data[t][0]['obj_bb'][camera_name][obj_name],
                                    obj_name)

                    for obj_name in obj_names:

                        if rescale_bb:
                            new_upper_left, new_bottom_right = scale_bb(
                                center=sample['traj']._data[t][0]['obj_bb'][
                                    camera_name][f"{obj_name}"]['center'],
                                upper_left=sample['traj']._data[t][0]['obj_bb'][
                                    camera_name][f"{obj_name}"]['upper_left_corner'],
                                bottom_right=sample['traj']._data[t][0]['obj_bb'][camera_name][f"{obj_name}"]['bottom_right_corner'])
                            sample['traj']._data[t][0]['obj_bb'][camera_name][f"{obj_name}"]['upper_left_corner'] = copy.deepcopy(
                                new_upper_left)
                            sample['traj']._data[t][0]['obj_bb'][camera_name][f"{obj_name}"]['bottom_right_corner'] = copy.deepcopy(
                                new_bottom_right)

                        # create last bb frame
                        sample['traj']._data[t][0]['obj_bb'][camera_name][f"{obj_name}_final"] = dict(
                        )
                        sample['traj']._data[t][0]['obj_bb'][camera_name][f"{obj_name}_final"][
                            'center'] = last_bb_all_obj[obj_name]['center']

                        if rescale_bb:
                            new_upper_left, new_bottom_right = scale_bb(
                                center=last_bb_all_obj[obj_name]['center'],
                                upper_left=last_bb_all_obj[obj_name]['upper_left_corner'],
                                bottom_right=last_bb_all_obj[obj_name]['bottom_right_corner'])
                        else:
                            new_upper_left, new_bottom_right = last_bb_all_obj[obj_name][
                                'upper_left_corner'], last_bb_all_obj[obj_name]['bottom_right_corner']

                        sample['traj']._data[t][0]['obj_bb'][camera_name][f"{obj_name}_final"]['upper_left_corner'] = copy.deepcopy(
                            new_upper_left)
                        sample['traj']._data[t][0]['obj_bb'][camera_name][f"{obj_name}_final"]['bottom_right_corner'] = copy.deepcopy(
                            new_bottom_right)

                        image = cv2.rectangle(sample['traj'].get(
                            t)['obs'][f"{camera_name}_image"].copy(),
                            sample['traj']._data[t][0]['obj_bb'][camera_name][
                                f"{obj_name}"]['upper_left_corner'],
                            sample['traj']._data[t][0]['obj_bb'][camera_name][
                                f"{obj_name}"]['bottom_right_corner'],
                            color=(0, 0, 255),
                            thickness=1)
                        image = cv2.rectangle(image,
                                              sample['traj']._data[t][0]['obj_bb'][camera_name][
                                                  f"{obj_name}_final"]['upper_left_corner'],
                                              sample['traj']._data[t][0]['obj_bb'][camera_name][
                                                  f"{obj_name}_final"]['bottom_right_corner'],
                                              color=(255, 0, 0),
                                              thickness=1)
                        image = cv2.circle(image,
                                           sample['traj']._data[t][0]['obj_bb'][
                                               camera_name][f"{obj_name}"]['center'],
                                           radius=1,
                                           color=(0, 0, 255),
                                           thickness=1)
                        image = cv2.circle(image,
                                           sample['traj']._data[t][0]['obj_bb'][
                                               camera_name][f"{obj_name}_final"]['center'],
                                           radius=1,
                                           color=(255, 0, 0),
                                           thickness=1)
                        cv2.imwrite(f"prova_bin_points_{obj_name}.jpg", image)

        if "real" in pkl_file_path or args.real:
            gripper = sample['traj'].get(t)['action'][-1]
            if start_pick_t == 0 and gripper == 1.0:
                start_pick_t = t

    if ("real" in pkl_file_path or args.real) and "task_00" in pkl_file_path:
        sampled_trj = list()
        sampled_trj.extend(sample['traj']._data[:1])
        sampled_trj.extend(sample['traj']._data[1:start_pick_t:5])
        sampled_trj.extend(sample['traj']._data[start_pick_t:-1:5])
        sampled_trj.extend(sample['traj']._data[-1:])
        sample['traj']._data = sampled_trj
        sample['len'] = len(sampled_trj)

    if False:  # "real" in pkl_file_path or args.real:
        # perform reshape a priori
        for t in range(len(sample['traj'])):
            for camera_name in ["camera_front", "camera_lateral_left", "camera_lateral_right", "eye_in_hand"]:
                img = sample['traj'].get(t)['obs'].get(
                    f"{camera_name}_image", None)
                if img is not None:
                    cv2.imwrite("original.png", img)
                    bb_dict = sample['traj'].get(
                        t)['obs'].get("obj_bb", None)
                    bb = None
                    if bb_dict is not None:
                        bb = bb_dict.get(camera_name, None)
                    img_res, adj_bb = crop_resize_img(task_cfg=task_spec,
                                                      task_name=task_name,
                                                      obs=img,
                                                      bb=bb
                                                      )
                    sample['traj'].get(
                        t)['obs'][f"{camera_name}_image"] = img_res
                    sample['traj'].get(
                        t)['obs']['obj_bb'][camera_name] = adj_bb
                    sample['traj'].get(
                        t)['obs']["target-object"] = int(int(sample['task_id'])/4)
                    img = copy.deepcopy(img_res)
                    for obj_name in adj_bb:
                        img = cv2.rectangle(img, adj_bb[obj_name]['upper_left_corner'],
                                            adj_bb[obj_name]['bottom_right_corner'],
                                            (0, 255, 0),
                                            1)
                    cv2.imwrite("prova.png", img)
                    # print("prova image")

    trj_name = pkl_file_path.split('/')[-1]
    out_pkl_file_path = os.path.join(out_path, trj_name)
    with open(out_pkl_file_path, "wb") as f:
        print(out_pkl_file_path)
        pickle.dump(sample, f)


if __name__ == '__main__':
    import debugpy
    # debugpy.listen(('0.0.0.0', 5678))
    # print("Waiting for debugger attach")
    # debugpy.wait_for_client()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="/", help="Path to task")
    parser.add_argument('--task_name', default="/", help="Name of the task")
    parser.add_argument('--robot_name', default="/", help="Name of the robot")
    parser.add_argument('--out_path', default=None,)
    parser.add_argument('--rescale_bb', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--real', action='store_true')

    args = parser.parse_args()

    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    # 1. Load the dataset
    folder_path = os.path.join(
        args.dataset_path, args.task_name, f"{args.robot_name}_{args.task_name}")
    # folder_path = "/user/frosa/multi_task_lfd/ur_multitask_dataset/pick_place/real_ur5e_pick_place/reduced_space/"
    if args.out_path is None:
        out_path = os.path.join(args.dataset_path,
                                f"{args.task_name}_opt",
                                f"{args.robot_name}_{args.task_name}")
    else:
        out_path = os.path.join(args.out_path,
                                f"{args.task_name}_opt",
                                f"{args.robot_name}_{args.task_name}")

    os.makedirs(name=out_path, exist_ok=True)

    # load task configuration file
    if args.real:
        conf_file_path = "../../experiments/tasks_cfgs/7_tasks_real.yaml"
    else:
        conf_file_path = "../../experiments/tasks_cfgs/7_tasks.yaml"
    with open(conf_file_path, 'r') as file:
        task_conf = yaml.safe_load(file)

    for dir in os.listdir(folder_path):
        # print(dir)
        if "task_" in dir:
            print(f"Considering task {dir}")
            out_task = os.path.join(out_path, dir)

            os.makedirs(name=out_task, exist_ok=True)

            task_path = os.path.join(folder_path, dir)

            i = 0
            trj_list = glob.glob(f"{task_path}/*.pkl")

            with Pool(WORKERS) as p:
                f = functools.partial(opt_traj,
                                      args.task_name,
                                      task_conf,
                                      out_task,
                                      args.rescale_bb,
                                      args.real
                                      )
                p.map(f, trj_list)
