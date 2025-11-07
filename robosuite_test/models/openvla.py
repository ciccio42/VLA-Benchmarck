import logging
import sys
from dataclasses import dataclass
import sys
from collections import deque
import os
import wandb
from .configs import OpenVLAConfig
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Union
import tensorflow as tf
import time
import cv2 
import tensorflow as tf
from robosuite.utils.transform_utils import quat2mat, mat2euler, euler2mat, quat2axisangle, mat2quat
import time
from PIL import Image
import random
import copy

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../../.")
from openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
    get_vla_action,
    check_unnorm_key,
    get_model,
    euler_to_axis_angle,
    SCALE_FACTOR,
    R_EE_TO_GRIPPER
)
from robot_utils import (
    DATE_TIME,
    TaskSuite
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK
from robosuite_utils import TASK_CROP, normalize_angle


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
    

def validate_config(cfg: OpenVLAConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


class open_vla_policy:

    def __init__(self, cfg: OpenVLAConfig):
        
        """Initialize model and associated components."""
        print("Initializing openvla model...")
        # Load model
        self.cfg = cfg
        model = get_model(cfg)

        # Load proprio projector if needed
        proprio_projector = None
        if cfg.use_proprio:
            proprio_projector = get_proprio_projector(
                cfg,
                model.llm_dim,
                proprio_dim=cfg.proprio_dim,  # 8-dimensional proprio for LIBERO
            )

        # Load action head if needed
        action_head = None
        if cfg.use_l1_regression or cfg.use_diffusion:
            action_head = get_action_head(cfg, model.llm_dim)

        # Load noisy action projector if using diffusion
        noisy_action_projector = None
        if cfg.use_diffusion:
            noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

        # Get OpenVLA processor if needed
        processor = None
        if isinstance(cfg, OpenVLAConfig):
            processor = get_processor(cfg)
            check_unnorm_key(cfg, model)

        self.model = model
        self.action_head = action_head
        self.proprio_projector = proprio_projector
        self.noisy_action_projector = noisy_action_projector
        self.processor = processor


    def compute_action(self, obs, resize_size, gripper_closed, task_description, task_name='pick_place', n_steps=-1):
        elapsed_time = 0
        start = time.time()
        # Prepare observation
        image = obs['camera_front_image']
        crop_params = TASK_CROP[task_name]
        
        top, left = crop_params[0], crop_params[2]
        img_height, img_width = image.shape[0], image.shape[1]
        box_h, box_w = img_height - top - crop_params[1], img_width - left - crop_params[3]
            
        # Crop the image
        cropped_image = image[top:top+box_h, left:left+box_w]

        # Optionally resize (define your desired size, e.g., 224x224)
        desired_height, desired_width = 224, 224
        resized_image = cv2.resize(cropped_image, (desired_width, desired_height), interpolation=cv2.INTER_LINEAR)
        obs['camera_front_image'] = resized_image
        
        observation, img = self.prepare_observation(obs=obs, 
                                                    resize_size=resize_size, gripper_closed=gripper_closed)
        
        action = None
        
        action = self.get_action(
                    obs = observation,
                    task_label = task_description,
                    )
        
        end = time.time()
        elapsed_time = end - start
        
        # perform action post-processing
        action = self.action_post_processing(obs=obs, 
                                             action_chunk=action, 
                                             n_steps=n_steps)


        return action, elapsed_time
    
    def prepare_observation(self, obs, resize_size, gripper_closed=0):
        img = obs['camera_front_image']
        eye_in_hand = cv2.flip(obs['eye_in_hand_image'], 1)  # Flip the image horizontally
        if isinstance(resize_size, int):
            resize_size = (resize_size, resize_size)
            
        # Resize using the same pipeline as in RLDS dataset builder
        img = tf.image.encode_jpeg(img)  # Encode as JPEG
        img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Decode back
        img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
        img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
        img = img.numpy()
        
        eye_in_hand = tf.image.encode_jpeg(eye_in_hand)  # Encode as JPEG
        eye_in_hand = tf.io.decode_image(eye_in_hand, expand_animations=False, dtype=tf.uint8)  # Decode back
        eye_in_hand = tf.image.resize(eye_in_hand, resize_size, method="lanczos3", antialias=True)
        eye_in_hand = tf.cast(tf.clip_by_value(tf.round(eye_in_hand), 0, 255), tf.uint8)
        eye_in_hand = eye_in_hand.numpy()

        eef_pose = np.zeros(6, dtype=np.float64)
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
        eef_pose = np.array(eef_pose, dtype=np.float64)


        # Prepare observations dict
        if gripper_closed == 0:
            gripper_closed = np.array([0.0, 0.0], dtype=np.float64)
        else:
            gripper_closed = np.array([0.0, 1.0], dtype=np.float64)
        
        state = np.zeros(8, dtype=np.float64)
        state[0:6] = eef_pose
        state[6:8] = gripper_closed
        observation = {
            "full_image": img,
            "camera_gripper_image": eye_in_hand,
            'state': state,
            'EEF_pose': eef_pose,
            'gripper_closed':gripper_closed, 
        }
        
        return observation, img


    def get_action(
        self,
        obs: Dict[str, Any],
        task_label: str,
        ) -> Union[List[np.ndarray], np.ndarray]:
            """
            Query the model to get action predictions.

            Args:
                cfg: Configuration object with model parameters
                model: The loaded model
                obs: Observation dictionary
                task_label: Text description of the task
                processor: Model processor for inputs
                action_head: Optional action head for continuous actions
                proprio_projector: Optional proprioception projector
                noisy_action_projector: Optional noisy action projector for diffusion
                use_film: Whether to use FiLM

            Returns:
                Union[List[np.ndarray], np.ndarray]: Predicted actions

            Raises:
                ValueError: If model family is not supported
            """
            with torch.no_grad():
                if isinstance(self.cfg, OpenVLAConfig):
                    action = get_vla_action(
                        cfg=self.cfg,
                        vla=self.model,
                        processor=self.processor,
                        obs=obs,
                        task_label=task_label,
                        action_head=self.action_head,
                        proprio_projector=self.proprio_projector,
                        noisy_action_projector=self.noisy_action_projector,
                        use_film=self.cfg.use_film,
                    )
                else:
                    raise ValueError(f"Unsupported model family: {type(self.cfg)}")

            return action

    def action_post_processing(self, obs, action_chunk=None, n_steps=-1):
        post_processed_actions = []
        for action in action_chunk:
            action = action*SCALE_FACTOR
            print(f"Action delta at time {n_steps}: {action}")
            
            # get current gripper position
            action_world = np.zeros(7)
            if 'abs_pose' in self.cfg.task_suite_name:
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
                
            post_processed_actions.append(copy.deepcopy(action_world))

def setup_logging(cfg: OpenVLAConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id

def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()

