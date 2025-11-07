from typing import Optional, Union
from pathlib import Path
import draccus
from dataclasses import dataclass
import sys
# Choice Registry lets you define a choice of implementations that can be selected at runtime
# Choice Registry lets you define a choice of implementations that can be selected at runtime
@dataclass
class ModelConfig(draccus.ChoiceRegistry):
    pass


@ModelConfig.register_subclass('openvla')
@dataclass
class OpenVLAConfig(ModelConfig):
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # Utils
    #################################################################################################################
    proprio_dim: int = 6
    # fmt: on
    chunk_size: int = 8         # Chunk size (number of actions to output at each policy query)
    task_suite_name: str = ''
    

@ModelConfig.register_subclass('tinyvla')
@dataclass
class TinyVLAConfig(ModelConfig):
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_path: str = "/home/rsofnc000/checkpoint_save_folder/tiny_vla/post-processed/checkpoint-3000"                   # Path to the model
    model_base: str = "/home/rsofnc000/checkpoint_save_folder/tiny_vla/post-processed/Llava-Pythia-1.3B"                    #
    enable_lora: bool = True                          # Whether to enable LoRA weights
    conv_mode: str = 'pythia'
    action_head: str = 'droid_diffusion'
    task_suite_name: str = '' 
    
    
@dataclass
class EvalConfig:
    
    model_family: str = "openvla"                    # Model family
    
    model_config: ModelConfig = OpenVLAConfig()  # Model configuration
    
    task_suite_name: str = "ur5e_pick_place_rm_central_spawn"  # Task suite
    
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 10                    # Number of rollouts per task
    
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)
    
    save: bool = True                             # Whether to save the trajectory and info

    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 42                                    # Random Seed (for reproducibility)
    
    controller_path: str = "/home/rsofnc000/Multi-Task-LFD-Framework/repo/openvla-oft/experiments/robot/robosuite/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json" # Path to custom controller config
    
    debug: bool = False                           # Whether to run in debug mode (for debugging purposes)
    run_number: int = 0                                  # Run number (for logging purposes)
    
    change_spawn_regions: bool = False             # Whether to change spawn regions for the tasks
    change_command: bool = False                   # Whether to change the command for the tasks
    object_set: int = -1                       # Whether to change the object for the tasks


