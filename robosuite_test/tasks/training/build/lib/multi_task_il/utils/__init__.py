import numpy as np
import torch
import copy
from collections import deque
from multi_task_il.datasets.savers import Trajectory
from torchvision.transforms import ToTensor, Normalize
import cv2
import hydra
import random
import os
from torchvision.transforms import ToTensor, Normalize
from torchvision.transforms.functional import resized_crop

# NORMALIZATION_RANGES = np.array([[-0.35,  0.35],
#                                  [-0.35,  0.35],
#                                  [0.60,  1.20],
#                                  [-3.14,  3.14911766],
#                                  [-3.14911766, 3.14911766],
#                                  [-3.14911766,  3.14911766]])


def normalize_action(action, n_action_bin, action_ranges, continous=False):
    half_action_bin = int(n_action_bin/2)
    norm_action = action.copy()
    # normalize between [-1 , 1]
    if action.shape[0] == 7:
        norm_action[:-1] = (2 * (norm_action[:-1] - action_ranges[:, 0]) /
                            (action_ranges[:, 1] - action_ranges[:, 0])) - 1

    else:
        norm_action = (2 * (norm_action - action_ranges[:, 0]) /
                       (action_ranges[:, 1] - action_ranges[:, 0])) - 1
    if continous:
        return norm_action
    else:
        # action discretization
        return (norm_action * half_action_bin).astype(np.int32).astype(np.float32) / half_action_bin


def denormalize_action(norm_action, action_ranges):
    action = np.clip(norm_action.copy(), -1, 1)
    for d in range(action_ranges.shape[0]):
        action[d] = (0.5 * (action[d] + 1) *
                     (action_ranges[d, 1] - action_ranges[d, 0])) + action_ranges[d, 0]
    return action


def denormalize_action_vima(norm_action, action_ranges):
    action = np.clip(norm_action.copy(), 0, 1)
    for d in range(action_ranges.shape[0]):
        action[d] = ((action[d]) *
                     (action_ranges[d, 1] - action_ranges[d, 0])) + action_ranges[d, 0]
    return action


def discretize_action(action, n_action_bin, action_ranges):
    disc_action = action.copy()
    # normalize between [0 , 1]
    disc_action[:-1] = ((disc_action[:-1] - action_ranges[:, 0]) /
                        (action_ranges[:, 1] - action_ranges[:, 0]))
    # if disc_action[-1] == -1:
    #     disc_action[-1] = 0
    # else:
    #     disc_action[-1] = 1

    return (disc_action * n_action_bin).astype(np.int32)


def load_trajectories(conf_file, mode='train'):
    conf_file.dataset_cfg.mode = 'train'
    return hydra.utils.instantiate(conf_file.dataset_cfg)


def select_random_frames(frames, n_select, sample_sides=True, experiment_number=1):
    selected_frames = []
    if experiment_number != 5:
        def clip(x): return int(max(0, min(x, len(frames) - 1)))
        per_bracket = max(len(frames) / n_select, 1)
        for i in range(n_select):
            n = clip(np.random.randint(
                int(i * per_bracket), int((i + 1) * per_bracket)))
            if sample_sides and i == n_select - 1:
                n = len(frames) - 1
            elif sample_sides and i == 0:
                n = 0
            selected_frames.append(n)
    elif experiment_number == 5:
        for i in range(n_select):
            # get first frame
            if i == 0:
                n = 0
            # get the last frame
            elif i == n_select - 1:
                n = len(frames) - 1
            elif i == 1:
                obj_in_hand = 0
                # get the first frame with obj_in_hand and the gripper is closed
                for t in range(1, len(frames)):
                    state = frames.get(t)['info']['status']
                    trj_t = frames.get(t)
                    gripper_act = trj_t['action'][-1]
                    if state == 'obj_in_hand' and gripper_act == 1:
                        obj_in_hand = t
                        n = t
                        break
            elif i == 2:
                # get the middle moving frame
                start_moving = 0
                end_moving = 0
                for t in range(obj_in_hand, len(frames)):
                    state = frames.get(t)['info']['status']
                    if state == 'moving' and start_moving == 0:
                        start_moving = t
                    elif state != 'moving' and start_moving != 0 and end_moving == 0:
                        end_moving = t
                        break
                n = start_moving + int((end_moving-start_moving)/2)

            selected_frames.append(n)

    if isinstance(frames, (list, tuple)):
        return [frames[i] for i in selected_frames]
    elif isinstance(frames, Trajectory):
        return [frames[i]['obs']['camera_front_image'] for i in selected_frames]
        # return [frames[i]['obs']['image-state'] for i in selected_frames]
    return frames[selected_frames]


def build_tvf_formatter(config, env_name='stack'):
    """Use this for torchvision.transforms in multi-task dataset, 
    note eval_fn always feeds in traj['obs']['images'], i.e. shape (h,w,3)
    """
    dataset_cfg = config.train_cfg.dataset
    height, width = dataset_cfg.get(
        'height', 100), dataset_cfg.get('width', 180)
    task_spec = config.get(env_name, dict())
    crop_params = task_spec.get('crop', [0, 0, 0, 0])
    top, left = crop_params[0], crop_params[2]

    def resize_crop(img):
        if len(img.shape) == 4:
            img = img[0]
        img_h, img_w = img.shape[0], img.shape[1]
        assert img_h != 3 and img_w != 3, img.shape
        box_h, box_w = img_h - top - \
            crop_params[1], img_w - left - crop_params[3]

        img = img.copy()
        obs = ToTensor()(img)
        obs = resized_crop(obs, top=top, left=left, height=box_h, width=box_w,
                           size=(height, width))

        obs = Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])(obs)

        return obs
    return resize_crop


def create_env(env_fn, agent_name, variation, ret_env, seed=None, heights=100, widths=180, gpu_id=0):
    from robosuite import load_controller_config
    
    if seed is None:
        create_seed = random.Random(None)
        create_seed = create_seed.getrandbits(32)
    else:
        create_seed = seed
    print(f"Creating environment with variation {variation}")

    # Load custom controller
    current_dir = os.path.dirname(os.path.abspath(__file__))
    controller_config_path = os.path.join(
        current_dir, "osc_pose.json")
    print(f"Controller path {controller_config_path}")
    controller = load_controller_config(
        custom_fpath=controller_config_path)
    return env_fn(agent_name, controller_type=controller, task=variation, ret_env=True, seed=create_seed, gpu_id=gpu_id)


def startup_env(model, env, context, gpu_id, variation_id, baseline=None, seed=None):
    done, states, images = False, [], []
    if baseline is None:
        states = deque(states, maxlen=1)
        images = deque(images, maxlen=1)  # NOTE: always use only one frame
    context = context.cuda(gpu_id)
    # np.random.seed(seed)
    while True:
        try:
            obs = env.reset()
            # action = np.zeros(7)
            # obs, _, _, _ = env.step(action)
            break
        except:
            pass
    traj = Trajectory()
    traj.append(obs)
    tasks = {'success': False, 'reached': False,
             'picked': False, 'variation_id': variation_id}
    return done, states, images, context, obs, traj, tasks


def torch_to_numpy(original_tensor):
    tensor = copy.deepcopy(original_tensor)
    tensor = Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225],
                       std=[1/0.229, 1/0.224, 1/0.225])(tensor)
    tensor = torch.mul(tensor, 255)
    # convert the tensor to a numpy array
    numpy_array = tensor.cpu().numpy()
    # transpose the numpy array to [y,h,w,c]
    numpy_array_transposed = np.transpose(
        numpy_array, (1, 3, 4, 2, 0))[:, :, :, :, 0]
    return numpy_array_transposed


def init_env(env, traj, task_name):
    # get objects id
    if task_name == 'pick_place':
        for obj_name in env.object_to_id.keys():
            obj = env.objects[env.object_to_id[obj_name]]
            # set object position based on trajectory file
            obj_pos = traj[1]['obs'][f"{obj_name}_pos"]
            obj_quat = traj[1]['obs'][f"{obj_name}_quat"]
            env.sim.data.set_joint_qpos(
                obj.joints[0], np.concatenate([obj_pos, obj_quat]))
    elif task_name == 'nut_assembly':
        for obj_name in env.env.nut_to_id.keys():
            obj = env.env.nuts[env.env.nut_to_id[obj_name]]
            obj_id = env.env.nut_to_id[obj_name]
            if obj_id == 0:
                obj_pos = traj[1]['obs']['round-nut_pos']
                obj_quat = traj[1]['obs']['round-nut_quat']
            else:
                obj_pos = traj[1]['obs'][f'round-nut-{obj_id+1}_pos']
                obj_quat = traj[1]['obs'][f'round-nut-{obj_id+1}_quat']
            # set object position based on trajectory file
            env.sim.data.set_joint_qpos(
                obj.joints[0], np.concatenate([obj_pos, obj_quat]))


# def get_action(model, states, images, context, gpu_id, n_steps, max_T=80, baseline=None):
#     s_t = torch.from_numpy(np.concatenate(states, 0).astype(np.float32))[None]
#     if isinstance(images[-1], np.ndarray):
#         i_t = torch.from_numpy(np.concatenate(
#             images, 0).astype(np.float32))[None]
#     else:
#         i_t = images[0][None]
#     s_t, i_t = s_t.cuda(gpu_id), i_t.cuda(gpu_id).float()

#     if baseline == 'maml':
#         learner = model.clone()
#         learner.adapt(
#             learner(None, context[0], learned_loss=True)['learned_loss'], allow_nograd=True, allow_unused=True)
#         out = learner(states=s_t[0], images=i_t[0], ret_dist=True)
#         action = out['action_dist'].sample()[-1].cpu().detach().numpy()

#     else:
#         with torch.no_grad():
#             out = model(states=s_t, images=i_t, context=context,
#                         eval=True)  # to avoid computing ATC loss
#             action = out['bc_distrib'].sample()[0, -1].cpu().numpy()
#             action = denormalize_action(
#                 norm_action=action, action_ranges=NORMALIZATION_RANGES)
#     # if TASK_NAME == 'nut_assembly':
#     #     action[3:7] = [1.0, 1.0, 0.0, 0.0]
#     action[-1] = 1 if action[-1] > 0 and n_steps < max_T - 1 else -1
#     return action


def load_model(model_path=None, step=0, conf_file=None):
    if model_path:
        # 1. Create the model starting from configuration
        model = hydra.utils.instantiate(conf_file.policy)
        # 2. Load weights
        weights = torch.load(os.path.join(
            model_path, f"model_save-{step}.pt"), map_location=torch.device('cpu'))
        model.load_state_dict(weights)
        return model
    else:
        raise ValueError("Model path cannot be None")
