import random
from train_utils import *
import torch
import hydra
import os

os.environ['MASTER_ADDR'] = 'localhost' 
os.environ['MASTER_PORT'] = '9959' 
torch.autograd.set_detect_anomaly(True)
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()


def seed_everything(seed=42):
    print(f"Cuda available {torch.cuda.is_available()}")
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@hydra.main(
    version_base=None,
    config_path="../experiments",
    config_name="config.yaml")
def main(cfg):
    if cfg.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    seed_everything(seed=42)

    from train_any import Workspace as W
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
            print(f"Number task for {tsk.name} {len(tsk.task_ids)}")
        cfg.bsize = sum(
            [(len(tsk.task_ids)-len(getattr(tsk, "skip_ids", []))) * cfg.set_same_n for tsk in cfg.tasks])
        cfg.vsize = cfg.bsize
        print(f"Computed batch-size {cfg.bsize}")
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

    if 'multi_task_il' not in cfg.policy._target_:
        print(f'Running baseline method: {cfg.policy._target_}')
        cfg.train_cfg.target_update_freq = -1

    workspace = W(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
