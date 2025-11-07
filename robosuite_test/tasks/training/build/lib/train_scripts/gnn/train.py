import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import hydra

from colorama import init as colorama_init
from colorama import Fore
from colorama import Style

colorama_init()


@hydra.main(
    version_base=None,
    config_path="../../experiments",
    config_name="config_gnn.yaml")
def main(cfg):
    if cfg.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    from workspace import Workspace as W
    all_tasks_cfgs = [cfg.tasks_cfgs.nut_assembly, cfg.tasks_cfgs.door, cfg.tasks_cfgs.drawer,
                      cfg.tasks_cfgs.button, cfg.tasks_cfgs.pick_place, cfg.tasks_cfgs.stack_block, cfg.tasks_cfgs.basketball]

    if cfg.task_names:
        cfg.tasks = [
            tsk for tsk in all_tasks_cfgs if tsk.name in cfg.task_names]

    if cfg.use_all_tasks:
        print("Loading all 7 tasks to the dataset!  obs_T: {} demo_T: {}".format(
            cfg.dataset_cfg.obs_T, cfg.dataset_cfg.demo_T))
        cfg.tasks = all_tasks_cfgs

    if cfg.exclude_task:
        print(f"Training with 6 tasks and exclude {cfg.exclude_task}")
        cfg.tasks = [
            tsk for tsk in all_tasks_cfgs if tsk.name != cfg.exclude_task]

    if cfg.set_same_n > -1:
        for tsk in cfg.tasks:
            tsk.n_per_task = cfg.set_same_n
        cfg.bsize = sum(
            [(len(tsk.task_ids)-len(getattr(tsk, "skip_ids", []))) * cfg.set_same_n for tsk in cfg.tasks])
        cfg.vsize = cfg.bsize
        print(
            f'To construct a training batch, set n_per_task of all tasks to {cfg.set_same_n}, new train/val batch sizes: {cfg.train_cfg.batch_size}/{cfg.train_cfg.val_size}')

    if cfg.limit_num_traj > -1:
        print('Only using {} trajectory for each sub-task'.format(cfg.limit_num_traj))
        for tsk in cfg.tasks:
            tsk.traj_per_subtask = cfg.limit_num_traj
    if cfg.limit_num_demo > -1:
        print(
            'Only using {} demon. trajectory for each sub-task'.format(cfg.limit_num_demo))
        for tsk in cfg.tasks:
            tsk.demo_per_subtask = cfg.limit_num_demo

    print(f"{Fore.GREEN}Running policy {cfg.policy}{Style.RESET_ALL}")

    workspace = W(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
