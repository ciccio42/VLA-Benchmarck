import json
import os
import shutil
import sys
import draccus
import numpy as np
import pickle as pkl
import random

# import Configuration File
sys.path.append('../.')
sys.path.append("/home/rsofnc000/Multi-Task-LFD-Framework/repo/TinyVLA")
sys.path.append("/home/rsofnc000/Multi-Task-LFD-Framework/repo/TinyVLA/test/robosuite_test/robosuite")
from robosuite_utils import *
from robot_utils import set_seed_everywhere, setup_logging, get_image_resize_size, TASK_VARIATION_DICT, COMMAND, TASK_MAX_STEPS
from robosuite_test.models.configs import EvalConfig


@draccus.wrap()
def eval_robosuite(cfg: EvalConfig) -> float:
    
    if cfg.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
    
    print("### Evaluating model ###")
    print(f"{cfg}")
    
    # Set random seed
    # if 'rm_12_13_14_15' in cfg.task_suite_name:
    #     set_seed_everywhere(0)
    # else:    
    
    set_seed_everywhere(cfg.seed)
    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)
    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)
    
    # Initialize Models
    run_episode_fn = None
    if cfg.model_family.lower() == "openvla":
        from robosuite_test.models.openvla import open_vla_policy
        # Validate configuration
        print(f"Running OpenVLA evaluation....")
        policy = open_vla_policy(cfg.model_config)
    elif cfg.model_family.lower() == "tinyvla":
        from robosuite_test.models.tinyvla import llava_pythia_act_policy
        print(f"Running TinyVLA evaluation....")
        policy = llava_pythia_act_policy(cfg.model_config)
        #run_tinyvla_eval(cfg)
    
    
    # # Initialize Robosuite environment
    # np.random.seed(42) # 42
    # random.seed(42)
    success_cnt = 0
    reached_cnt = 0
    picked_cnt = 0
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"seeds/seeds.txt"), 'r') as f:
        lines = f.readlines()
        seeds = [int(line.strip()) for line in lines]
    
    test_variations = TASK_VARIATION_DICT[cfg.task_suite_name]
    
    for ctr in range(cfg.num_trials_per_task*len(test_variations)):
        variation_id = test_variations[ctr % len(test_variations)]
        env_seed = seeds[ctr % len(seeds)]
        gpu_id = 0
        
        if 'pick_place' in cfg.task_suite_name:
            env_name = 'pick_place'
            
        # save trajectory and info
        save_path = os.path.join(cfg.model_config.model_path, f"rollout_{env_name}_{cfg.run_number}_{cfg.change_spawn_regions}_obj_set_{cfg.object_set}_change_command_{cfg.change_command}")#_task_list_{test_variations}")
        os.makedirs(save_path, exist_ok=True)
        print(f"Saving rollout to: {save_path}")
        if os.path.exists(os.path.join(save_path, f"info_{ctr}.json")):
            print(f"Trajectory {ctr} already exists, skipping...")
            
            # load results json file
            with open(os.path.join(save_path, f"info_{ctr}.json"), "r") as f:
                info = json.load(f)
                success_cnt += info['success']
                reached_cnt += info['reached']
                picked_cnt += info['picked']
                print(f"Reached rate: {reached_cnt / (ctr + 1):.2f}")
                print(f"Picked rate: {picked_cnt / (ctr + 1):.2f}")
                print(f"Success rate: {success_cnt / (ctr + 1):.2f}")
            continue
            
        env = build_env_context(env_name=env_name,
                        controller_path=cfg.controller_path,
                        variation=variation_id,
                        seed=env_seed,
                        gpu_id=gpu_id,
                        object_set=cfg.object_set)
        eval_fn = get_eval_fn(env_name=env_name)
        
        if cfg.object_set == -1:
            task_description = COMMAND[env_name][str(variation_id)]
        else:
            task_description = COMMAND[f"{env_name}_{cfg.object_set}"][str(variation_id)]
            

        if cfg.change_command:
            if 'red' in task_description:
                task_description = task_description.replace('red', 'orange')
        print(f"Running Task Description: {task_description}")
        
        # set_seed_everywhere(cfg.seed)
        # remove saved images
        # shutil.rmtree("./images/", ignore_errors=True)
        traj, info = eval_fn(cfg = cfg,  
                            policy = policy, 
                            env = env, 
                            variation_id = variation_id, 
                            max_T = TASK_MAX_STEPS[cfg.task_suite_name], 
                            resize_size = resize_size, 
                            task_description= task_description,
                            task_name = env_name,
                            change_spawn_regions=cfg.change_spawn_regions)
    
        
        print("Evaluated traj #{}, task#{}, reached? {} picked? {} success? {} ".format(ctr, variation_id, info['reached'], info['picked'], info['success']))
        success_cnt += info['success']
        reached_cnt += info['reached']
        picked_cnt += info['picked']
        
        traj._data[0][0]['task_description'] = task_description
        
        if cfg.save:
            pkl.dump(traj, open(os.path.join(save_path, f"traj_{ctr}.pkl"), "wb"))
        
        # Save info
        with open(os.path.join(save_path, f"info_{ctr}.json"), "w") as f:
            json.dump(info, f, indent=4)
        
        print(f"Reached rate: {reached_cnt / (ctr + 1):.2f}")
        print(f"Picked rate: {picked_cnt / (ctr + 1):.2f}")
        print(f"Success rate: {success_cnt / (ctr + 1):.2f}")
        

    
    print(f"Reached rate: {reached_cnt / ctr:.2f}")
    print(f"Picked rate: {picked_cnt / ctr:.2f}")
    print(f"Success rate: {success_cnt / ctr:.2f}")
    


if __name__ == "__main__":
    eval_robosuite()
