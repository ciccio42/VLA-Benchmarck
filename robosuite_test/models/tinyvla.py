import torch
import numpy as np
import os
import pickle
import cv2
from robosuite.utils.transform_utils import quat2mat, mat2euler, euler2mat, quat2axisangle, mat2quat
from torchvision import transforms
import time
import copy

from llava_pythia.model.language_model.pythia.llava_pythia import LlavaPythiaConfig
from llava_pythia.conversation import conv_templates, SeparatorStyle
from llava_pythia.model.builder import load_pretrained_model
from llava_pythia.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava_pythia.model.language_model.pythia.llava_pythia import LlavaPythiaConfig
from llava_pythia.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from robosuite_utils import normalize_angle, TASK_CROP
from robot_utils import set_seed_everywhere
from tinyvla_utils import rot_6d_to_euler_angles, SCALE_FACTOR, R_EE_TO_GRIPPER, euler_to_axis_angle
from PIL import Image
from torchvision.transforms.functional import to_pil_image

class llava_pythia_act_policy:
    """
    Policy class for Llava-Pythia action generation.

    Attributes:
        policy_config: Configuration dictionary for the policy.
    """
    def __init__(self, policy_config, data_args=None):
        super(llava_pythia_act_policy).__init__()
        self.load_policy(policy_config)
        self.data_args = data_args
        self.to_tensor = transforms.ToTensor()
        self.all_time_actions = None
        
        self.action_dim = self.policy.config.action_dim

        self.policy.eval()

        stats_path = os.path.join("/".join(self.policy_config.model_path.split('/')[:-1]), f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)

        if self.policy_config.action_head == 'act':
            self.post_process = lambda a: a * self.stats['action_std'] + self.stats['action_mean']
        elif self.policy_config.action_head == 'transformer_diffusion':
            self.post_process = lambda a: ((a + 1) / 2) * (self.stats['action_max'] - self.stats['action_min']) + self.stats['action_min']
        elif self.policy_config.action_head == 'droid_diffusion':
            self.post_process = lambda a: (((a + 1) / 2) * (self.stats['action_max'] - self.stats['action_min'])) + self.stats['action_min']


    def eval(self):
        self.policy.eval()

    def load_policy(self, policy_config):
        self.policy_config = policy_config
        # self.conv = conv_templates[policy_config['conv_mode']].copy()
        model_base = policy_config.model_base if policy_config.enable_lora else None
        model_name = get_model_name_from_path(policy_config.model_path)
        model_path = policy_config.model_path

        self.tokenizer, self.policy, self.image_processor, self.context_len = load_pretrained_model(model_path, model_base, model_name, False, False)
        self.config = LlavaPythiaConfig.from_pretrained('/'.join(model_path.split('/')[:-1]), trust_remote_code=False)

    def process_batch_to_llava(self, curr_image, robo_state, raw_lang):
        """
        Processes a batch of data for Llava-Pythia model input.

        Args:
            curr_image: Current image tensor.
            robo_state: Current robot state tensor.
            raw_lang: Raw language input.

        Returns:
            A dictionary containing processed data for the model.
        """
        self.conv = conv_templates[self.policy_config.conv_mode].copy()

        if len(curr_image.shape) == 5: # 1,2,3,270,480
            curr_image = curr_image.squeeze(0)

        # for k,v in sample.items():
        #     print(k, v.shape)
        image, image_r = torch.chunk(curr_image, 2, dim=0)

        image = self.expand2square(image, tuple(x for x in self.image_processor.image_mean))
        image_tensor = self.image_processor.preprocess(image, 
                                                       return_tensors='pt', 
                                                       do_normalize=True, 
                                                       do_rescale=False, 
                                                       do_center_crop=False)['pixel_values']
        image_tensor = image_tensor.to(self.policy.device, dtype=self.policy.dtype)

        image_r = self.expand2square(image_r, tuple(x for x in self.image_processor.image_mean))
        image_tensor_r = self.image_processor.preprocess(image_r, return_tensors='pt', do_normalize=True, do_rescale=False, do_center_crop=False)['pixel_values']
        image_tensor_r = image_tensor_r.to(self.policy.device, dtype=self.policy.dtype)

        # print('raw_lang')
        inp = raw_lang #.decode('utf-8')
        assert image is not None, 'image must be provided.'
        # first message
        if self.policy.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        self.conv.append_message(self.conv.roles[0], inp)
        image = None

        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()
        prompt += " <|endoftext|>"

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        attn_mask = input_ids.ne(self.tokenizer.pad_token_id)
        states = robo_state.to(self.policy.device, dtype=self.policy.dtype)
        # print(input_ids.dtype, attn_mask.dtype, image_tensor.dtype, image_tensor_r.dtype, states.dtype)

        data_dict = dict(input_ids=input_ids,
                         attention_mask=attn_mask,
                         images=image_tensor,
                         images_r=image_tensor_r,
                         states=states.unsqueeze(0))  # Add batch dimension

        # print(f"@@@@@@@@@@@@@@@{image_tensor.shape}")
        return data_dict

    def expand2square(self, pil_imgs, background_color):
        batch_size, channels, height, width = pil_imgs.shape
        max_dim = max(height, width)
        expanded_imgs = np.full((batch_size, max_dim, max_dim, channels), background_color, dtype=np.float32)

        if height == width:
            expanded_imgs = pil_imgs.permute(0,2,3,1).cpu().numpy()
        elif height > width:
            offset = (max_dim - width) // 2
            # expanded_imgs[:, :height, offset:offset + width] = pil_imgs
            expanded_imgs[:, :height, offset:offset + width, :] = pil_imgs.permute(0,2,3,1).cpu().numpy()
        else:
            offset = (max_dim - height) // 2
            # expanded_imgs[:, offset:offset + height, :width] = pil_imgs
            expanded_imgs[:, offset:offset + height, :width, :] = pil_imgs.permute(0,2,3,1).cpu().numpy()
        expanded_imgs = torch.tensor(expanded_imgs).to(dtype=pil_imgs.dtype, device=pil_imgs.device) # B H W C
        return expanded_imgs
    
    
    def compute_action(self, obs, resize_size, gripper_closed, task_description, task_name='pick_place', n_steps=-1):    
        
        if self.policy_config.action_head == 'act':
            rand_crop_resize = False
            temporal_agg = True
        else:
            rand_crop_resize = True
            temporal_agg = True        
    

        query_frequency = self.policy.config.chunk_size / 2 # specify the exact executed action steps, must be smaller than chunk size
        if temporal_agg:
            query_frequency = 1
            num_queries = self.policy.config.chunk_size
        else:
            query_frequency = 1
        max_timesteps = int(200)  # may increase for real-world tasks

        ### evaluation loop
        if temporal_agg and n_steps == 0:
            # reset all_time_actions if new episode
            self.all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, self.action_dim],dtype=torch.float32).cuda()
            # print(f'all_time_actions size: {all_time_actions.size()}')
        
        replay_traj = dict()
        image_list = []  # for visualization
        robot_state_list = []
        target_action_list = []
        success = False

        with torch.no_grad():  #torch.inference_mode():
            traj_rgb_np, robot_state = self.prepare_observation(
                                                obs=obs,
                                                stats=self.stats,
                                                task_name=task_name,
                                                gripper_closed=gripper_closed)
            
            # image_list.append(cv2.resize(traj_rgb_np[0], (224,224)))
            robot_state_list.append(robot_state)
            robot_state = torch.from_numpy(robot_state).float().cuda()
            
            if n_steps % query_frequency == 0:
                curr_image =  []
                for img in traj_rgb_np:
                    curr_image.append(self.to_tensor(img).float().cuda())
                curr_image = torch.stack(curr_image, dim=0)  # stack images along batch dimension
                # curr_image = to_tensor(traj_rgb_np).float().cuda() #torch.from_numpy(traj_rgb_np / 255.0).float().cuda()
                if rand_crop_resize:
                    # print('rand crop resize is used!')
                    original_size = curr_image.shape[-2:]
                    ratio = 0.95
                    curr_image = curr_image[:, :,
                                    int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                                    int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
                    curr_image = curr_image.squeeze(0)
                    resize_transform = transforms.Resize(original_size, antialias=True)
                    curr_image = resize_transform(curr_image)
                    curr_image = curr_image.unsqueeze(0)
                else:
                    curr_image = curr_image.unsqueeze(0)  # add batch dimension
                    
            if n_steps == 0:
                # warm up
                print('network warm up started')
                # set_seed_everywhere(0)
                for _ in range(10):
                    # ext_img = to_pil_image(curr_image[0,0])
                    # ext_img.save(f'./warmup_ext_img_{_}.png')
                    # gripper_img = to_pil_image(curr_image[0,1])
                    # gripper_img.save(f'./warmup_gripper_img_{_}.png')
                    batch = self.process_batch_to_llava(curr_image, robot_state, task_description)
                    self.policy(**batch, eval=True)
                print('network warm up done')
                time1 = time.time()
        
            
            ### query policy
            time3 = time.time()
            if self.policy_config.action_head == "act":
                if n_steps % query_frequency == 0:
                    batch = self.process_batch_to_llava(curr_image, robot_state, task_description)
                    all_actions = self.policy(**batch, eval=True)

                if temporal_agg:
                    # print(f"all_actions: {self.all_actions.size()}")
                    # print(f"all_time_actions: {self.all_time_actions.size()}")
                    # print(f"t: {t}, num_queries:{num_queries}")
                    self.all_time_actions[[n_steps], n_steps:n_steps + num_queries] = all_actions
                    actions_for_curr_step = self.all_time_actions[:, n_steps]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = all_actions[:, t % query_frequency]
            elif self.policy_config.action_head == "droid_diffusion":
                if n_steps % query_frequency == 0:
                    ext_img = to_pil_image(curr_image[0,0])
                    # ext_img.save(f'./images/ext_img_{n_steps}.png')
                    gripper_img = to_pil_image(curr_image[0,1])
                    # gripper_img.save(f'./images/gripper_img_{n_steps}.png')
                    batch = self.process_batch_to_llava(curr_image, robot_state, task_description)
                    all_actions = self.policy(**batch, eval=True)
                    inf_time = time.time() - time3
                if temporal_agg:
                    # print(f"all_actions: {all_actions.size()}")
                    # print(f"all_time_actions: {self.all_time_actions.size()}")
                    # print(f"n_steps: {n_steps}, num_queries:{num_queries}")
                    self.all_time_actions[[n_steps], n_steps:n_steps + num_queries] = all_actions
                    actions_for_curr_step = self.all_time_actions[:, n_steps]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    
                else:
                    raw_action = all_actions[0][0] #all_actions[:, n_steps % query_frequency]
            else:
                raise NotImplementedError

            # print(f"raw action size: {raw_action.size()}")
            ### post-process actions
            raw_action = raw_action.squeeze(0).cpu().numpy()
            action = raw_action
            action = self.post_process(raw_action)
            # print(f"after post_process action size: {action.shape}")
            
            
            action = self.convert_actions(action)
            # action = normalize_gripper_action(action=action, binarize=True)
            # action = invert_gripper_action(action)
            # print(f"after normalization and inversion: {action}")
            time5 = time.time()
            
            action = self.action_post_processing(obs=obs, 
                                                action_chunk=action[None, :], 
                                                n_steps=n_steps)
            
            return action, inf_time
        

    def prepare_observation(self, obs, task_name, stats, gripper_closed):
        """
        Retrieves observations (images and robot states) from the robot environment.

        Returns:
            A tuple containing images and states.
        """
        # Crop front image
        img = obs['camera_front_image']
        crop_params = TASK_CROP[task_name]
        
        top, left = crop_params[0], crop_params[2]
        img_height, img_width = img.shape[0], img.shape[1]
        box_h, box_w = img_height - top - crop_params[1], img_width - left - crop_params[3]
        # Crop the img
        cropped_image = img[top:top+box_h, left:left+box_w]

        # Optionally resize (define your desired size, e.g., 224x224)
        desired_height, desired_width = 224, 224
        resized_image = cv2.resize(cropped_image, (desired_width, desired_height))
        obs['camera_front_image'] = resized_image
        
        # Get in hand image
        eye_in_hand = cv2.flip(obs['eye_in_hand_image'], 1)  # Flip the image horizontally
        eye_in_hand = cv2.resize(eye_in_hand, (224, 224))
        
        images = np.array([cv2.resize(resized_image, (320, 180)),
                           cv2.resize(eye_in_hand, (320, 180))])
        images = images / 255.0  # Normalize to [0, 1] range
        
        eef_pose = np.zeros(6, dtype=np.float32)
        # convert gripper orientation to end effector orientation
        eef_quat = obs['eef_quat']
        R_ee_to_gripper = np.array([[.0, -1.0, .0], 
                                    [1.0, .0, .0], 
                                    [.0, .0, 1.0]])
        eef_mat = R_ee_to_gripper @ quat2mat(eef_quat)
        eef_euler = [normalize_angle(a) for a in mat2euler(eef_mat)]
        # print(f"EEF Euler State: {eef_euler}")
        eef_pose[0:3] = obs['eef_pos']
        eef_pose[3:6] = eef_euler
        eef_pose = np.array(eef_pose, dtype=np.float32)
        
        
        # Prepare observations dict
        if gripper_closed == 0:
            gripper_closed = np.array([0.0, 0.0], dtype=np.float32)
        else:
            gripper_closed = np.array([0.0, 1.0], dtype=np.float32)
        state = np.zeros(8, dtype=np.float32)
        state[0:6] = eef_pose
        state[6:8] = gripper_closed
        
        # normalize states
        state = (state - stats["qpos_mean"]) / stats["qpos_std"]
        return images, state


    def convert_actions(self,pred_action):
        # pred_action = torch.from_numpy(actions)
        # pred_action = actions.squeeze(0)
        cur_xyz = pred_action[:3]
        cur_rot6d = pred_action[3:9]
        cur_gripper = np.expand_dims(pred_action[-1], axis=0)

        cur_rot6d = torch.from_numpy(cur_rot6d).unsqueeze(0)
        cur_euler = rot_6d_to_euler_angles(rot_6d=cur_rot6d, convention="XYZ").squeeze().numpy()
        # print(f'cur_xyz size: {cur_xyz.shape}')
        # print(f'cur_euler size: {cur_euler.shape}')
        # print(f'cur_gripper size: {cur_gripper.shape}')
        pred_action = np.concatenate((cur_xyz, cur_euler, cur_gripper))
        # print(f'4. pred_action size: {pred_action.shape}')
        # print(f'4. after convert pred_action: {pred_action}')

        return pred_action
    
    def action_post_processing(self, obs, action_chunk=None, n_steps=-1):
        post_processed_actions = []
        for action in action_chunk:
            action = action*SCALE_FACTOR
            # print(f"Action delta at time {n_steps}: {action}")
            
            # get current gripper position
            action_world = np.zeros(7)
            if 'abs_pose' in self.policy_config.task_suite_name:
                action_world[0:3] = action[0:3]
            else:
                # Position action in world frame
                # action[0] = np.clip(action[0], 0.0, 1.0)
                # if action[6] >= 0.8 and action[0] < 0:
                #     action[0] = -action[0]
                
                action_world[0:3] = obs['eef_pos'] + action[0:3]
                
                # Orientation action in world frame
                current_gripper_orientation =  mat2euler(R_EE_TO_GRIPPER @ quat2mat(obs['eef_quat']))
                current_gripper_orientation =  [normalize_angle(a) for a in current_gripper_orientation]
                gripper_orientation_action = current_gripper_orientation + action[3:6]
                gripper_orientation_action = [normalize_angle(a) for a in gripper_orientation_action]
                action_world[3:6] = euler_to_axis_angle(gripper_orientation_action)
                action_world[6] = action[-1]
                
            post_processed_actions.append(copy.deepcopy(action_world))
        
        return post_processed_actions