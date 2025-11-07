import warnings
from multi_task_il.utils import *
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import hydra
import os
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import functools
from torch.multiprocessing import Pool, set_start_method
from multi_task_robosuite_env.controllers.controllers.expert_nut_assembly import \
    get_expert_trajectory as nut_expert
from multi_task_robosuite_env.controllers.controllers.expert_pick_place import \
    get_expert_trajectory as place_expert
import sys
import pickle as pkl
import json
import wandb
set_start_method('forkserver', force=True)
sys.path.append('/home/Multi-Task-LFD-Framework/repo/mosaic/tasks/test_models')
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# python dataset_analysis.py --model /home/ciccio/Desktop/multi_task_lfd/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/baseline-1/1Task-Pick-Place-Stable-Policy-Batch32-1gpu-Attn2ly128-Act2ly256mix4-headCat-simclr128x512 --step 72900 --task_indx 12 --debug

# To be crystal clear: each constructed "Environment" is defined by both (task_name, robot_name), e.g. 'PandaBasketball'
# but each task_name may have differnt sub-task ids: e.g. Basketball-task_00 is throwing the white ball into hoop #1
TASK_ENV_MAP = {
    'pick_place': {
        'n_task':   16,
        'env_fn':   place_expert,
        'eval_fn':  pick_place_eval,
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
    },
}


def single_run(model, agent_env, context, img_formatter, variation, show_image, agent_trj, indx):
    start, end = torch.cuda.Event(
        enable_timing=True), torch.cuda.Event(enable_timing=True)
    eval_fn = TASK_ENV_MAP[args.task_name]['eval_fn']
    with torch.no_grad():
        start.record()
        traj, info, context = eval_fn(model=model,
                                      env=agent_env,
                                      context=context,
                                      gpu_id=0,
                                      variation_id=variation,
                                      img_formatter=img_formatter,
                                      max_T=150,
                                      agent_traj=agent_trj,
                                      model_act=True,
                                      show_img=show_image)
        print("Evaluated traj #{}, task#{}, reached? {} picked? {} success? {} ".format(
            indx, variation, info['reached'], info['picked'], info['success']))
        end.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = start.elapsed_time(end)
        print(f"Elapsed time {curr_time}")

    return traj, info


def run_inference(model, conf_file, task_name, task_indx, results_dir_path, training_trj, show_image, experiment_number, file_pair):

    model.cuda(0)
    variation_number = task_indx if task_indx != -1 else int(file_pair[0][1])
    demo_file = file_pair[0][2]
    demo_file_name = demo_file.split('/')[-1]
    agent_file = file_pair[0][3]
    agent_file_name = agent_file.split('/')[-1]
    sample_number = file_pair[1]
    results_analysis = [task_name, variation_number,
                        sample_number, demo_file_name, agent_file_name]

    # print(f"----\nDemo file {demo_file}\nAgent file {agent_file}\n----")
    # open demo and agent data
    with open(demo_file, "rb") as f:
        demo_data = pickle.load(f)
    with open(agent_file, "rb") as f:
        agent_data = pickle.load(f)

    # get target object id
    demo_target = demo_data['traj'].get(0)['obs']['target-object']
    agent_target = agent_data['traj'].get(0)['obs']['target-object']
    if demo_target != agent_target:
        print(f"Sample indx {sample_number} different target objects")

    # get env function
    env_func = TASK_ENV_MAP[task_name]['env_fn']
    agent_name = TASK_ENV_MAP[task_name]['ur5e']
    variation = variation_number
    ret_env = True
    heights = conf_file.dataset_cfg.height
    widths = conf_file.dataset_cfg.width
    agent_env = create_env(env_fn=env_func, agent_name=agent_name,
                           variation=variation, ret_env=ret_env)

    img_formatter = build_tvf_formatter(conf_file, task_name)

    if training_trj:
        agent_trj = agent_data['traj']
    else:
        agent_trj = None

    if experiment_number == 1:
        cnt = 10
        np.random.seed(0)
    elif experiment_number == 5:
        cnt = 10
        np.random.seed(0)
    else:
        cnt = 1
    for i in range(cnt):
        # select context frames
        context = select_random_frames(
            demo_data['traj'], 4, sample_sides=True, experiment_number=experiment_number)
        # perform normalization on context frames
        context = [img_formatter(i[:, :, ::-1])[None] for i in context]
        if isinstance(context[0], np.ndarray):
            context = torch.from_numpy(
                np.concatenate(context, 0)).float()[None]
        else:
            context = torch.cat(context, dim=0).float()[None]

        traj, info = single_run(model=model,
                                agent_env=agent_env,
                                context=context,
                                img_formatter=img_formatter,
                                variation=variation, show_image=show_image, agent_trj=agent_trj, indx=variation_number)
        results_analysis.append(info)
        info['demo_file'] = demo_file_name
        info['agent_file'] = agent_file_name
        info['task_name'] = task_name
        pkl.dump(traj, open(results_dir_path +
                 '/traj{}_{}_{}.pkl'.format(variation_number, sample_number, i), 'wb'))
        pkl.dump(context, open(results_dir_path +
                 '/context{}_{}_{}.pkl'.format(variation_number, sample_number, i), 'wb'))
        res = {}
        for k, v in info.items():
            if v == True or v == False:
                res[k] = int(v)
            else:
                res[k] = v
        json.dump(res, open(results_dir_path +
                  '/traj{}_{}_{}.json'.format(variation_number, sample_number, i), 'w'))

    del model
    return results_analysis


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--step', type=int)
    parser.add_argument('--task_indx', type=int)
    parser.add_argument('--results_dir', type=str, default="/")
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--project_name', type=str, default=None)
    parser.add_argument('--task_name', type=str, default="pick_place")
    parser.add_argument('--experiment_number', type=int, default=1,
                        help="1: Take samples from list and run 10 times with different demonstrator frames; 2: Take all the file from the training set")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--training_trj', action='store_true')
    parser.add_argument('--show_img', action='store_true')
    parser.add_argument('--run_inference', action='store_true')

    args = parser.parse_args()

    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    # 1. Load Training Dataset
    conf_file = OmegaConf.load(os.path.join(args.model, "config.yaml"))
    # 2. Get the dataset
    dataset = load_trajectories(conf_file=conf_file, mode='train')
    # 3. Load model
    model = load_model(model_path=args.model,
                       step=args.step, conf_file=conf_file)
    model.eval()

    task_name = args.task_name

    # try all training-samples
    file_pairs = [(dataset.all_file_pairs[indx], indx)
                  for indx in range(len(dataset.all_file_pairs))]

    if args.experiment_number == 1 or args.experiment_number == 2 or args.experiment_number == 5:
        if args.project_name:
            model_name = f"{args.model.split('/')[-1]}-{args.step}"
            run = wandb.init(project=args.project_name,
                             job_type='test', group=model_name.split("-1gpu")[0])
            run.name = model_name + f'-Test_{model_name}-Step_{args.step}'
            wandb.config.update(args)

        if args.task_indx != -1:
            results_dir_path = os.path.join(
                args.results_dir, f"results_{task_name}", str(f"task-{args.task_indx}"), f"step_{args.step}")
        else:
            results_dir_path = os.path.join(
                args.results_dir, f"results_training_{task_name}", f"step_{args.step}")
        try:
            os.makedirs(results_dir_path)
        except:
            pass

        # model, conf_file, task_name, task_indx, results_dir_path, training_trj, show_image, file_pair
        f = functools.partial(run_inference, model, conf_file, task_name, args.task_indx,
                              results_dir_path, args.training_trj, args.show_img, args.experiment_number)

        if args.num_workers > 1:
            with Pool(args.num_workers) as p:
                task_success_flags = p.map(f, file_pairs)
        else:
            task_success_flags = [f(file_pair) for file_pair in file_pairs]
