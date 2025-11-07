# from multi_task_test.eval_functions import *
import os
import json
import sys
import cv2
import copy
# import multi_task_robosuite_env as mtre
from multi_task_robosuite_env.controllers.controllers.expert_nut_assembly import \
    get_expert_trajectory as nut_expert
from multi_task_robosuite_env.controllers.controllers.expert_pick_place import \
    get_expert_trajectory as place_expert
from multi_task_robosuite_env.controllers.controllers.expert_button import \
    get_expert_trajectory as button_expert
from multi_task_robosuite_env.controllers.controllers.expert_block_stacking import \
    get_expert_trajectory as stack_block_expert
import numpy as np
import robosuite.utils.transform_utils as T
from robosuite import load_controller_config
from collections import deque



TASK_CROP = {
    'pick_place': [20, 25, 80, 75],
    'nut_assembly': [20, 25, 80, 75],
    'stack_block': [20, 25, 80, 75],
    'press_button': [10, 10, 70, 70], 
}

ENV_OBJECTS = {
    'pick_place': {
        'obj_names': ['greenbox', 'yellowbox', 'bluebox', 'redbox', 'bin'],
        'bin_names': ['bin_box_1', 'bin_box_2', 'bin_box_3', 'bin_box_4'],
        'ranges': [[-0.255, -0.195], [-0.105, -0.045], [0.045, 0.105], [0.195, 0.255]],
        'splitted_obj_names': ['green box', 'yellow box', 'blue box', 'red box'],
        'bin_position': [0.18, 0.00, 0.75],
        'obj_dim': {'greenbox': [0.05, 0.055, 0.045],  # W, H, D
                    'yellowbox': [0.05, 0.055, 0.045],
                    'bluebox': [0.05, 0.055, 0.045],
                    'redbox': [0.05, 0.055, 0.045],
                    'bin': [0.6, 0.06, 0.15]},
    },
    'nut_assembly': {
        'obj_names': ['round-nut', 'round-nut-2', 'round-nut-3'],
        'peg_names': ['peg1', 'peg2', 'peg3'],
        'splitted_obj_names': ['grey nut', 'brown nut', 'blue nut'],
        'ranges': [[0.10, 0.31], [-0.10, 0.10], [-0.31, -0.10]]
    },
    'stack_block': {
        'obj_names': ['cubeA', 'cubeB', 'cubeC'],
    },
    'button': {
        'obj_names': ['machine1_goal1', 'machine1_goal2', 'machine1_goal3',
                      'machine2_goal1', 'machine2_goal2', 'machine2_goal3'],
        'place_names': ['machine1_goal1_final', 'machine1_goal2_final', 'machine1_goal3_final',
                        'machine2_goal1_final', 'machine2_goal2_final', 'machine2_goal3_final']
    }
}

TASK_MAP = {
    'nut_assembly':  {
        'num_variations':   9,
        'env_fn':   nut_expert,
        'agent-teacher': ('UR5e_NutAssemblyDistractor', 'Panda_NutAssemblyDistractor'),
        'render_hw': (200, 360),
        'object_set': 1,
    },
    'pick_place': {
        'num_variations':   16,
        'num_variations_per_object':   4,
        'env_fn':   place_expert,
        'agent-teacher': ('UR5e_PickPlaceDistractor', 'Panda_PickPlaceDistractor'),
        'render_hw': (200, 360),  # (150, 270)
        'object_set': 2,
    },
    'stack_block': {
        'num_variations':   6,
        'env_fn':   stack_block_expert,
        'agent-teacher': ('UR5e_BlockStacking', 'Panda_BlockStacking'),
        'render_hw': (200, 360),  # (150, 270)
        'object_set': 1,
    },
    'button': {
        'num_variations':   6,
        'env_fn':   button_expert,
        'agent-teacher': ('UR5e_Button', 'Panda_Button'),
        'render_hw': (200, 360),  # (150, 270)
        'object_set': 1,
    },
}

def normalize_angle(angle, tol=1e-1):
    """
    Normalize angle to (-π, π], where -π wraps to π
    """
    norm = (angle + np.pi) % (2 * np.pi) - np.pi
    if np.isclose(norm, -np.pi, atol=tol):
        norm = np.pi
    return norm

def build_env_context(env_name: str, controller_path: str, variation: int, seed: int, gpu_id: int, object_set: int):
    # load custom controller
    controller = load_controller_config(
        custom_fpath=controller_path)
    
    build_task = TASK_MAP.get(env_name, None)
    assert build_task, 'Got unsupported task '+env_name
    
    env_fn = build_task['env_fn']
    agent_name, teacher_name = build_task['agent-teacher']

    
    agent_env = env_fn(agent_name,
                       controller_type=controller,
                       task=variation,
                       ret_env=True,
                       seed=seed,
                       gpu_id=gpu_id,
                       object_set=TASK_MAP[env_name]['object_set'] if object_set == -1 else object_set)
    
    return agent_env
    
    
def get_eval_fn(env_name):

    sys.path.append(os.path.join(os.path.dirname(__file__), "test"))
    if "pick_place" in env_name:
        from test.pick_place import pick_place_eval
        return pick_place_eval
    elif "nut_assembly" in env_name:
        NotImplementedError
    elif "button" in env_name:
        NotImplementedError
    elif "stack" in env_name:
        NotImplementedError
    else:
        assert NotImplementedError


def startup_env(env, variation_id, spawn_region=None):

    done, states, images = False, [], []
    states = deque(states, maxlen=1)
    images = deque(images, maxlen=1)  # NOTE: always use only one frame
    
    while True:
        try:
            obs = env.reset()
            cv2.imwrite("pre_change.jpg", obs['camera_front_image'][:,:, ::-1])
           
            if spawn_region is not None:
                target_obj_name = env.objects[env.object_id].name.lower()
                target_obj_pos = obs[f"{target_obj_name}_pos"]
                target_obj_quat = obs[f"{target_obj_name}_quat"]
                # check if the object is not in the OOD spawn region
                if not(target_obj_pos[1] <= spawn_region[0] and target_obj_pos[1] >= spawn_region[1]):
                    print(f"Object {target_obj_name} is not in the OOD spawn region {spawn_region}, ")
                    
                    for obj_id in range(len(env.objects)):
                        if obj_id == env.object_id:
                            continue
                        object_name_to_change = env.objects[obj_id].name.lower()
                        object_pos = obs[f"{object_name_to_change}_pos"]
                        object_quat = obs[f"{object_name_to_change}_quat"]
                        
                        # check if the object is in the OOD spawn region
                        if object_pos[1] <= spawn_region[0] and object_pos[1] >= spawn_region[1]:
                            print(f"Object {object_name_to_change} is in the OOD spawn region {spawn_region}, ")
                            break
                        
                        
                    new_obj_pos =  obs[f"{object_name_to_change}_pos"]
                    new_obj_quat = obs[f"{object_name_to_change}_quat"]
                    
                    # set position of the target objects
                    env.sim.data.set_joint_qpos(env.objects[env.object_id].joints[0], 
                                                np.concatenate([new_obj_pos, new_obj_quat]))  

                    # set the position of the other object
                    env.sim.data.set_joint_qpos(env.objects[obj_id].joints[0],
                                                np.concatenate([target_obj_pos, target_obj_quat]))
                        

            cv2.imwrite("pre_set.jpg", obs['camera_front_image'][:,:, ::-1])
            # make a "null step" to stabilize all objects
            current_gripper_position = env.sim.data.site_xpos[env.robots[0].eef_site_id]
            current_gripper_orientation = T.quat2axisangle(T.mat2quat(np.reshape(
                env.sim.data.site_xmat[env.robots[0].eef_site_id], (3, 3))))
            current_gripper_pose = np.concatenate(
                (current_gripper_position, current_gripper_orientation, np.array([-1])), axis=-1)
            i = 0   
            while i < 5:
                obs, reward, env_done, info = env.step(current_gripper_pose)
                cv2.imwrite("post_set.jpg", obs['camera_front_image'][:,:, ::-1])
                i+=1
            break
        except:
            pass

    traj = Trajectory()
    traj.append(obs)
    tasks = {'success': False, 'reached': False,
             'picked': False, 'variation_id': variation_id}
    
    return done, states, images, obs, traj, tasks, current_gripper_pose


def check_pick(threshold: float, obj_z: float, start_z: float, reached: bool, picked: bool):
    return picked or (reached and abs(obj_z - start_z) > threshold)


def check_reach(threshold: float, obj_distance: np.array, current_reach: bool):
    return current_reach or np.linalg.norm(
        obj_distance) < threshold


def check_bin(threshold: float, bin_pos: np.array, obj_pos: np.array, current_bin: bool):
    bin_x_low = bin_pos[0]
    bin_y_low = bin_pos[1]
    bin_x_low -= 0.16 / 2
    bin_y_low -= 0.16 / 2

    bin_x_high = bin_x_low + 0.16
    bin_y_high = bin_y_low + 0.16
    # print(bin_pos, obj_pos)
    res = False
    if (
            bin_x_low < obj_pos[0] < bin_x_high
            and bin_y_low < obj_pos[1] < bin_y_high
            and bin_pos[2] < obj_pos[2] < bin_pos[2] + 0.1
    ):
        res = True
    return (current_bin or res)


def check_peg(peg_pos: np.array, obj_pos: np.array, current_peg: bool):

    # print(bin_pos, obj_pos)
    res = False
    if (
            abs(obj_pos[0] - peg_pos[0]) < 0.03
            and abs(obj_pos[1] - peg_pos[1]) < 0.03
            and obj_pos[2] < 0.860 + 0.05
    ):
        res = True
    return res or current_peg


#### -------- ---- ---- ---- ---- ---- ---- ---- ---- ####
#### Trajectory class
#### to store observations, actions, rewards, etc.
#### -------- ---- ---- ---- ---- ---- ---- ---- ---- ####


def _compress_obs(obs):
    for key in obs.keys():
        if 'image' in key:
            if len(obs[key].shape) == 3:
                okay, im_string = cv2.imencode('.jpg', obs[key])
                assert okay, "image encoding failed!"
                obs[key] = im_string
        if 'depth_norm' in key:
            assert len(
                obs[key].shape) == 2 and obs[key].dtype == np.uint8, "assumes uint8 greyscale depth image!"
            depth_im = np.tile(obs[key][:, :, None], (1, 1, 3))
            okay, depth_string = cv2.imencode('.jpg', depth_im)
            assert okay, "depth encoding failed!"
            obs[key] = depth_string
    return obs


def _decompress_obs(obs):
    keys = ["camera_front_image"]
    for key in keys:
        if 'image' in key:
            try:
                decomp = cv2.imdecode(obs[key], cv2.IMREAD_COLOR)
                obs[key] = decomp
            except:
                pass
        if 'depth_norm' in key:
            obs[key] = cv2.imdecode(
                obs[key], cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    return obs


class Trajectory:
    def __init__(self, config_str=None):
        self._data = []
        self._raw_state = []
        self.set_config_str(config_str)

    def append(self, obs, reward=None, done=None, info=None, action=None, raw_state=None):
        """
        Logs observation and rewards taken by environment as well as action taken
        """
        obs, reward, done, info, action, raw_state = [copy.deepcopy(
            x) for x in [obs, reward, done, info, action, raw_state]]

        obs = _compress_obs(obs)
        self._data.append((obs, reward, done, info, action))
        self._raw_state.append(raw_state)

    @ property
    def T(self):
        """
        Returns number of states
        """
        return len(self._data)

    def __getitem__(self, t):
        return self.get(t)

    def get(self, t, decompress=True):
        assert 0 <= t < self.T or - \
            self.T < t <= 0, "index should be in (-T, T)"

        obs_t, reward_t, done_t, info_t, action_t = self._data[t]
        if decompress:
            obs_t = _decompress_obs(obs_t)
        ret_dict = dict(obs=obs_t, reward=reward_t,
                        done=done_t, info=info_t, action=action_t)

        for k in list(ret_dict.keys()):
            if ret_dict[k] is None:
                ret_dict.pop(k)
        return ret_dict

    def change_obs(self, t, obs):
        obs_t, reward_t, done_t, info_t, action_t = self._data[t]
        self._data[t] = obs, reward_t, done_t, info_t, action_t

    def __len__(self):
        return self.T

    def __iter__(self):
        for d in range(self.T):
            yield self.get(d)

    def get_raw_state(self, t):
        assert 0 <= t < self.T or - \
            self.T < t <= 0, "index should be in (-T, T)"
        return copy.deepcopy(self._raw_state[t])

    def set_config_str(self, config_str):
        self._config_str = config_str

    @ property
    def config_str(self):
        return self._config_str

