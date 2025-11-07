from robosuite import load_controller_config
from multi_task_robosuite_env.controllers.controllers.expert_pick_place import \
    get_expert_trajectory as place_expert
from multi_task_robosuite_env.controllers.controllers.expert_nut_assembly import \
    get_expert_trajectory as nut_expert
from multi_task_robosuite_env.controllers.controllers.expert_block_stacking import \
    get_expert_trajectory as block_stacking_expert
from multi_task_robosuite_env.controllers.controllers.expert_button import \
    get_expert_trajectory as button_expert
import functools
import os
import pickle as pkl
import json
import random
import torch
from os.path import join
from multiprocessing import Pool, cpu_count

# To be crystal clear: each constructed "Environment" is defined by both (task_name, robot_name), e.g. 'PandaBasketball'
# but each task_name may have differnt sub-task ids: e.g. Basketball-task_00 is throwing the white ball into hoop #1
TASK_ENV_MAP = {
    'pick_place': {
        'n_task':   16,
        'env_fn':   place_expert,
        'panda':    'Panda_PickPlaceDistractor',
        'sawyer':   'Sawyer_PickPlaceDistractor',
        'ur5e':     'UR5e_PickPlaceDistractor',
        'object_set': 2,
    },
    'nut_assembly':  {
        'n_task':   9,
        'env_fn':   nut_expert,
        'panda':    'Panda_NutAssemblyDistractor',
        'sawyer':   'Sawyer_NutAssemblyDistractor',
        'ur5e':     'UR5e_NutAssemblyDistractor',
        'object_set': 1,
    },
    'block_stacking':  {
        'n_task':   6,
        'env_fn':   block_stacking_expert,
        'panda':    'Panda_BlockStacking',
        'sawyer':   'Sawyer_BlockStacking',
        'ur5e':     'UR5e_BlockStacking',
        'object_set': 1,
    },
    'press_button':  {
        'n_task':   6,
        'env_fn':   button_expert,
        'panda':    'Panda_Button',
        'sawyer':   'Sawyer_Button',
        'ur5e':     'UR5e_Button',
        'object_set': 1,
    },
}

ROBOT_NAMES = ['panda', 'sawyer', 'ur5e']

# open command json file
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "command.json")) as f:
    TASK_COMMAND = json.load(f)


def save_rollout(N, task_name, env_type, env_func, save_dir, n_tasks, env_seed=False, camera_obs=True, seeds=None, n_per_group=1, ctrl_config='IK_POSE', renderer=False, gpu_count=1, gpu_id_indx=-1, color=False, shape=False):
    if isinstance(N, int):
        N = [N]
    for n in N:
        # NOTE(Mandi): removed the 'continue' part, always writes new data
        task = int((n % (n_tasks * n_per_group)) // n_per_group)
        print(task)
        seed = None if seeds is None else seeds[n]
        env_seed = seeds[n - n %
                         n_per_group] if seeds is not None and env_seed else None
        if ctrl_config == 'IK_POSE' or ctrl_config == 'OSC_POSE':
            config = load_controller_config(default_controller=ctrl_config)
        else:
            config = load_controller_config(custom_fpath=ctrl_config)
        if gpu_id_indx == -1:
            gpu_id_indx = int(n % gpu_count)
        if color or shape:
            assert 'BlockStacking' in env_type, env_type
            traj = env_func(env_type, controller_type=config, renderer=renderer,
                            camera_obs=camera_obs, task=task,
                            seed=seed, env_seed=env_seed, gpu_id=gpu_id_indx, color=color, shape=shape)
        else:
            traj = env_func(env_type,
                            controller_type=config,
                            renderer=renderer,
                            camera_obs=camera_obs,
                            task=task,
                            seed=seed,
                            env_seed=env_seed,
                            gpu_id=gpu_id_indx,
                            render_camera="camera_front",
                            object_set=TASK_ENV_MAP[task_name]['object_set'])
            if len(traj) < 5:  # try again
                traj = env_func(env_type, controller_type=config, renderer=renderer, camera_obs=camera_obs, task=task,
                                seed=seed, env_seed=env_seed, gpu_id=gpu_id_indx, render_camera="camera_front")

        # let's make a new folder structure for easier dataloader construct:
        # env_type/task_id/traj_idx.pkl, where idxes start from 0 within each sub-task
        group_count = n // (n_tasks * n_per_group)
        traj_idx = n % n_per_group + n_per_group * group_count

        save_path = os.path.join(save_dir, 'task_{:02d}'.format(task))
        os.makedirs(save_path, exist_ok=1)
        file_name = os.path.join(save_path, 'traj{:03d}.pkl'.format(traj_idx))

        if task_name == "pick_place":
            command_key = f"{task_name}_set_{TASK_ENV_MAP['pick_place']['object_set']}"
        else:
            command_key = task_name

        pkl.dump({
            'traj': traj,
            'len': len(traj),
            'env_type': env_type,
            'command': TASK_COMMAND[command_key][str(task)],
            'task_id': task}, open(file_name, 'wb'))
        del traj


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('save_dir', default='./',
                        help='Folder to save rollouts')
    parser.add_argument('--num_workers', default=cpu_count(), type=int,
                        help='Number of collection workers (default=n_cores)')
    parser.add_argument('--ctrl_config', type=str,
                        help="Path to controller configuration file")
    parser.add_argument('--N', default=10, type=int,
                        help="Number of trajectories to collect")
    parser.add_argument('--per_task_group', default=100, type=int,
                        help="Number of trajectories of same task in row")
    # NOTE(Mandi): these are new:
    parser.add_argument('--task_name', '-tsk', default='nut',
                        type=str, help="Environment name")
    parser.add_argument('--robot', '-ro', default='panda',
                        type=str, help="Robot name")
    parser.add_argument('--overwrite', action='store_true',
                        help="Carefully overwrite stuff only when specified")
    parser.add_argument('--object_set', default=1, type=int)

    parser.add_argument('--collect_cam', action='store_true',
                        help="If flag then will collect camera observation")
    parser.add_argument('--renderer', action='store_true',
                        help="If flag then will display rendering GUI")
    parser.add_argument('--random_seed', action='store_true',
                        help="If flag then will collect data from random envs")
    parser.add_argument('--n_env', default=None, type=int,
                        help="Number of environments to collect from")
    parser.add_argument('--n_tasks', default=12, type=int,
                        help="Number of tasks in environment")
    parser.add_argument('--gpu_id_indx', default=-1,
                        type=int, help="GPU to use for rendering")
    parser.add_argument('--give_env_seed', action='store_true',
                        help="Maintain seperate consistent environment sampling seed (for multi obj envs)")

    # for blockstacking only:
    parser.add_argument('--color', action='store_true')
    parser.add_argument('--shape', action='store_true')

    parser.add_argument('--debugger', action='store_true')

    args = parser.parse_args()

    if args.debugger:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    assert args.num_workers > 0, "num_workers must be positive!"

    if args.random_seed:
        assert args.n_env is None
        seeds = [None for _ in range(args.N)]
    elif args.n_env:
        envs, rng = [263237945 +
                     i for i in range(args.n_env)], random.Random(385008283)
        seeds = [int(rng.choice(envs)) for _ in range(args.N)]
    else:
        n_per_group = args.per_task_group
        seeds = [263237945 + int(n // (args.n_tasks * n_per_group))
                 * n_per_group + n % n_per_group for n in range(args.N)]
    # select proper names and functions
    assert (
        args.task_name in args.save_dir and args.robot in args.save_dir), args.save_dir

    if args.task_name == "pick_place":
        TASK_ENV_MAP['pick_place']['object_set'] = args.object_set

    assert args.task_name in TASK_ENV_MAP.keys(
    ), 'Got unsupported task. name {}'.format(args.task_name)
    print("Collecting {} trajs for {}, using {} subtasks".format(
        args.N, args.task_name, args.n_tasks))
    print('Saving path: ', args.save_dir)

    assert args.robot in ROBOT_NAMES, 'Got unsupported robot name {}'.format(
        args.robot)
    specs = TASK_ENV_MAP.get(args.task_name)
    env_name = specs.get(args.robot, None)
    env_fn = specs.get('env_fn', None)
    assert env_name and env_fn, env_name+'is unsupported'
    print("Making environment {} for robot {}, using env builder {}".format(
        env_name, args.robot, env_fn))

    # handle path info
    if not os.path.exists(args.save_dir):
        assert args.overwrite, "Make sure don't overwrite existing data unintendedly."
        os.makedirs(args.save_dir, exist_ok=1)
    assert os.path.isdir(
        args.save_dir), "directory specified but is a file and not directory! " + args.save_dir
    os.makedirs(args.save_dir, exist_ok=1)

    json.dump(
        {
            'robot':    args.robot,
            'task':     args.task_name,
            'env_type': env_name,
            'n_tasks':  args.n_tasks,
            'task_group_size': args.per_task_group,
        },
        open(join(args.save_dir, 'info.json'), 'w'))

    count = torch.cuda.device_count()
    print("Distributing work to %s GPUs" % count)
    if args.num_workers == 1:
        save_rollout(
            N=list(range(args.N)),
            task_name=args.task_name,
            env_type=env_name, env_func=env_fn, save_dir=args.save_dir, n_tasks=args.n_tasks,
            env_seed=args.give_env_seed, camera_obs=args.collect_cam, seeds=seeds, n_per_group=args.per_task_group,
            renderer=args.renderer,
            gpu_count=count,
            gpu_id_indx=args.gpu_id_indx,
            ctrl_config=args.ctrl_config,
            color=args.color, shape=args.shape)
    else:
        assert not args.renderer, "can't display rendering when using multiple workers"

        with Pool(args.num_workers) as p:
            f = functools.partial(
                save_rollout,
                env_type=env_name,
                task_name=args.task_name, env_func=env_fn, save_dir=args.save_dir, n_tasks=args.n_tasks,
                env_seed=args.give_env_seed, camera_obs=args.collect_cam, seeds=seeds, n_per_group=args.per_task_group,
                renderer=args.renderer,
                gpu_count=count,
                gpu_id_indx=args.gpu_id_indx,
                ctrl_config=args.ctrl_config,
                color=args.color, shape=args.shape)

            p.map(f, range(args.N))
