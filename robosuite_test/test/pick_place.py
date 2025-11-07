from robosuite_utils import startup_env, check_reach, check_pick, check_bin, ENV_OBJECTS 
import numpy as np
from PIL import Image
from robosuite.utils.transform_utils import quat2mat, mat2euler, euler2mat, quat2axisangle, mat2quat
import robosuite.utils.transform_utils as T
from robosuite_utils import normalize_angle
import os
from robot_utils import set_seed_everywhere

OBJECT_SET = 2
SCALE_FACTOR = 0.05 # SCale factor for actions
Y_SPAWN_REGION = [[0.255, 0.195], [0.105, 0.045], [-0.045, -0.105], [-0.195, -0.255]]
SPAWN_REGION_REMOVED_PER_OBJ = {
    'greenbox': [0.255, 0.195],
    'yellowbox': [0.105, 0.045],
    'bluebox': [-0.045, -0.105],
    'redbox': [-0.195, -0.255]
}

def pick_place_eval(cfg, policy, env, variation_id, max_T, resize_size, task_description: str,
                    task_name = 'pick_place', change_spawn_regions=False):

    if change_spawn_regions and cfg.task_suite_name == "ur5e_pick_place_removed_spawn_regions":
        print(f"Using spawn regions for variation {variation_id}")
        # spawn objects in regions not seen during training
        spawn_region = SPAWN_REGION_REMOVED_PER_OBJ[env.objects[env.object_id].name.lower()]
    elif not change_spawn_regions and cfg.task_suite_name == "ur5e_pick_place_removed_spawn_regions":
        # sample a Training Spawn Region
        spawn_indx = np.random.randint(0, len(Y_SPAWN_REGION))
        spawn_region = Y_SPAWN_REGION[spawn_indx]
        while spawn_region == SPAWN_REGION_REMOVED_PER_OBJ[env.objects[env.object_id].name.lower()]:
            spawn_indx = np.random.randint(0, len(Y_SPAWN_REGION))
            spawn_region = Y_SPAWN_REGION[spawn_indx]
        
    elif change_spawn_regions and cfg.task_suite_name == "ur5e_pick_place_rm_one_spawn":
        print(f"Using spawn regions for variation {variation_id}")
        # spawn objects in regions not seen during training
        spawn_region = [0.255, 0.195]  # only one spawn region for all objects
    elif not change_spawn_regions and cfg.task_suite_name == "ur5e_pick_place_rm_one_spawn":
        # sample a Training Spawn Region
        spawn_indx = np.random.randint(0, len(Y_SPAWN_REGION))
        spawn_region = Y_SPAWN_REGION[spawn_indx]
        while spawn_region == [0.255, 0.195]:
            spawn_indx = np.random.randint(0, len(Y_SPAWN_REGION))
            spawn_region = Y_SPAWN_REGION[spawn_indx]
    
    elif change_spawn_regions and cfg.task_suite_name == "ur5e_pick_place_rm_central_spawn":
        print(f"Using spawn regions for variation {variation_id}")
        # spawn objects in regions not seen during training
        spawn_region = [0.105, 0.045] # only one spawn region for all objects
    elif not change_spawn_regions and cfg.task_suite_name == "ur5e_pick_place_rm_central_spawn":
        # sample a Training Spawn Region
        spawn_indx = np.random.randint(0, len(Y_SPAWN_REGION))
        spawn_region = Y_SPAWN_REGION[spawn_indx]
        while spawn_region == [0.105, 0.045]:
            spawn_indx = np.random.randint(0, len(Y_SPAWN_REGION))
            spawn_region = Y_SPAWN_REGION[spawn_indx]        
 
    else:
        print(f"Using default spawn regions for variation {variation_id}")
        # spawn objects in default regions
        spawn_region = None

    start_up_env_return = startup_env(
                        env=env,
                        variation_id=variation_id,
                        spawn_region=spawn_region,
                        )
    done, states, images, obs, traj, tasks, current_gripper_pose = start_up_env_return
    
    img = Image.fromarray(obs['camera_front_image'])
    # img.save("first_frame.jpg")
    
    
    n_steps = 0

    object_name_target = env.objects[env.object_id].name.lower()
    obj_delta_key = object_name_target + '_to_robot0_eef_pos'
    obj_key = object_name_target + '_pos'
    start_z = obs[obj_key][2]
    
    
    print(f"Max-t {max_T}")
    tasks["reached_wrong"] = 0.0
    tasks["picked_wrong"] = 0.0
    tasks["place_wrong"] = 0.0
    tasks["place_wrong_correct_obj"] = 0.0
    tasks["place_wrong_wrong_obj"] = 0.0
    tasks["place_correct_bin_wrong_obj"] = 0.0
    elapsed_time = 0.0
    
    # os.makedirs("images", exist_ok=True)
    
    previous_action = np.zeros((7,), dtype=np.float32)
    
    while not done:

        tasks['reached'] = int(check_reach(threshold=0.04,
                                           obj_distance=obs[obj_delta_key][:2],
                                           current_reach=tasks['reached']
                                           ))

        tasks['picked'] = int(check_pick(threshold=0.05,
                                         obj_z=obs[obj_key][2],
                                         start_z=start_z,
                                         reached=tasks['reached'],
                                         picked=tasks['picked']))

        for obj_id, obj_name, in enumerate(env.env.obj_names):
            if obj_id != traj.get(0)['obs']['target-object'] and obj_name != "bin":
                if check_reach(threshold=0.04,
                                obj_distance=obs[obj_name.lower() +
                                                '_to_robot0_eef_pos'],
                                current_reach=tasks.get(
                                    "reached_wrong", 0.0)
                                ):
                    tasks['reached_wrong'] = 1.0
                if check_pick(threshold=0.04,
                                obj_z=obs[obj_name.lower() + "_pos"][2],
                                start_z=start_z,
                                reached=tasks['reached_wrong'],
                                picked=tasks.get(
                                    "picked_wrong", 0.0)):
                    tasks['picked_wrong'] = 1.0

        if n_steps == 0:
            gripper_closed = 0.0
        else:
            gripper_closed = 0 if action_world[6] == -1.0 else 1.0

        action_world_chunk,  time_action = policy.compute_action(
                                obs = obs,
                                resize_size = resize_size,
                                gripper_closed = gripper_closed,
                                task_description = task_description,
                                task_name = task_name,
                                n_steps = n_steps)

        for action_world in action_world_chunk:
            # print(f"\n---- Predicted gripper {action_world[6]} ----")
            if not gripper_closed and round(action_world[6], 2) > 0.9:
                # action_world[2] = action[2] - 0.05
                action_world[6] = 1.0
            elif not gripper_closed and round(action_world[6], 2) < 0.9:
                action_world[6] = -1.0
            elif gripper_closed and round(action_world[6], 2) < 0.7:
                action_world[6] = -1.0
            elif gripper_closed and round(action_world[6], 2) > 0.7:
                action_world[6] = 1.0

            
            
                
            # avoid too strong gripper orientation changes
            if n_steps > 0:
                previous_gripper_orientation_action = previous_action[3:6]
                current_gripper_orientation_action = action_world[3:6]
                
                # check if the gripper state changed
                if action_world[6] != previous_action[6]:
                    gripper_state_changed = True
                else:
                    gripper_state_changed = False
                    
                previous_action[6] = action_world[6]
                
                if (abs(previous_gripper_orientation_action[0] - current_gripper_orientation_action[0]) > 2.0 or abs(previous_gripper_orientation_action[1] - current_gripper_orientation_action[1]) > 2.0 or abs(previous_gripper_orientation_action[2] - current_gripper_orientation_action[2]) > 2.0):
                    action_world[3:6] = previous_gripper_orientation_action
                else:
                    previous_action = action_world.copy()
                    
            else:
                gripper_state_changed = False
                previous_action = action_world.copy()

            try:
                # current_gripper_orientation = T.quat2axisangle(T.mat2quat(np.reshape(
                # env.sim.data.site_xmat[env.robots[0].eef_site_id], (3, 3))))
                # action_world[3:6] = current_gripper_orientation
                # print(f"\nAction delta {action}\nAction world before step {n_steps}: {action_world}")
                # print(f"Predicted gripper state: {action[6]}")
                n_steps += 1
                if gripper_state_changed:
                    if action_world[6] == 1.0:
                        action_world[6] = -1.0
                        action_world[2] -= 0.03  # move down before closing the gripper
                        obs, reward, env_done, info = env.step(action_world)
                        action_world[6] = 1.0
                        obs, reward, env_done, info = env.step(action_world)
                        for i in range(10):
                            obs, reward, env_done, info = env.step(action_world)
                        # action_world[2] += 0.05  # move up after closing the gripper
                        
                    else:
                        action_world[6] = 1.0
                        obs, reward, env_done, info = env.step(action_world)
                        action_world[6] = -1.0
                        # action_world[6] = 1.0
                        obs, reward, env_done, info = env.step(action_world)
                else:
                    obs, reward, env_done, info = env.step(action_world)
                    #print(f"Reward: {reward}, Env done: {env_done}")
                    # print(f"\tPose after step {n_steps}: {obs['eef_pos']}")    
                
                if reward == 1:
                    print("Success!")
                elif env_done:
                    print(f"Episode finished! Env done: {env_done}")
            except:
                print("Episode finished! Raised an exception")
                done = True
                tasks['success'] = 0
                break
        
            image_step = obs['camera_front_image']
            image_step = Image.fromarray(image_step)
            # image_step.save("step.jpg")

            # save the combination of gripper image and camera front image
            gripper_img = obs['robot0_eye_in_hand_image']
            gripper_img = Image.fromarray(gripper_img)
            # gripper_img.save("gripper_step.jpg")
            
            # stack the gripper image and camera front image
            image = Image.new('RGB', (gripper_img.width + image_step.width, gripper_img.height))
            image.paste(gripper_img, (0, 0))
            image.paste(image_step, (gripper_img.width, 0))
            # save the combined image
            # image.save(f"images/step_{n_steps}.jpg")
            
            # current_step = obs['camera_front_image']
            # img = Image.fromarray(current_step)
            # img.save(f"current_step.jpg")
            
            
            traj.append(obs, reward, done, info, action_world)
            elapsed_time += time_action
            
            tasks['success'] = int(reward or tasks['success'])
        
        
            # check if the object has been placed in a different bin
            if not tasks['success']:
                for i, bin_name in enumerate(ENV_OBJECTS['pick_place']['bin_names']):
                    if i != obs['target-box-id']:
                        bin_pos = obs[f"{bin_name}_pos"]
                        if check_bin(threshold=0.03,
                                        bin_pos=bin_pos,
                                        obj_pos=obs[f"{object_name_target}_pos"],
                                        current_bin=tasks.get(
                                            "place_wrong_correct_obj", 0.0)
                                        ):
                            tasks["place_wrong_correct_obj"] = 1.0

                for obj_id, obj_name, in enumerate(env.env.obj_names):
                    if obj_id != traj.get(0)['obs']['target-object'] and obj_name != "bin":
                        for i, bin_name in enumerate(ENV_OBJECTS['pick_place']['bin_names']):
                            if i != obs['target-box-id']:
                                bin_pos = obs[f"{bin_name}_pos"]
                                if check_bin(threshold=0.03,
                                                bin_pos=bin_pos,
                                                obj_pos=obs[f"{obj_name}_pos"],
                                                current_bin=tasks.get(
                                                    "place_wrong_wrong_obj", 0.0)
                                                ):
                                    tasks["place_wrong_wrong_obj"] = 1.0
                            else:
                                bin_pos = obs[f"{bin_name}_pos"]
                                if check_bin(threshold=0.03,
                                                bin_pos=bin_pos,
                                                obj_pos=obs[f"{obj_name}_pos"],
                                                current_bin=tasks.get(
                                                    "place_correct_bin_wrong_obj", 0.0)
                                                ):
                                    tasks["place_correct_bin_wrong_obj"] = 1.0

            
            if env_done or reward or n_steps >= max_T-1 or tasks["place_correct_bin_wrong_obj"] == 1 or tasks["place_wrong_wrong_obj"] == 1 or tasks["place_wrong_correct_obj"] == 1:
                done = True
                break

            
    print(tasks)
    env.close()
    mean_elapsed_time = elapsed_time/n_steps
    print(f"Mean elapsed time {mean_elapsed_time}")
    
    del env
    del states
    del images
    del policy
    del obs

    return traj, tasks