import os
import random
from multiprocessing import cpu_count

import numpy as np
import torch
from hydra.utils import instantiate
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from multi_task_il_gnn.datasets.batch_sampler import BatchSampler
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
from collections import OrderedDict
import time
from collections import defaultdict
import logging

colorama_init()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')

# Create a logger object
logger = logging.getLogger('Data-Loader')


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_by_task(batch):
    """ Use this for validation: groups data by task names to compute per-task losses """
    collate_time = time.time()
    per_task_data = defaultdict(list)
    start_batch = time.time()
    for b in batch:
        per_task_data[b['task_name']].append(
            {k: v for k, v in b.items() if k != 'task_name' and k != 'task_id'}
        )
    logger.debug(f"Batch time {time.time()-start_batch}")

    collate_time = time.time()
    for name, data in per_task_data.items():
        per_task_data[name] = default_collate(data)
    logger.debug(f"Collate time {time.time()-collate_time}")
    return per_task_data


def make_data_loaders(config, dataset_cfg):

    print(f"{Fore.GREEN}Creating Trainign dataset{Style.RESET_ALL}")

    dataset_cfg.mode = "train"
    dataset = instantiate(dataset_cfg)

    train_step = int(config.get('epochs') *
                     int(len(dataset)/config.get('bsize')))
    print(f"{Fore.GREEN}Number of train/step {train_step}{Style.RESET_ALL}")

    samplerClass = BatchSampler
    train_sampler = samplerClass(
        task_to_idx=dataset.task_to_idx,
        subtask_to_idx=dataset.subtask_to_idx,
        tasks_spec=dataset_cfg.tasks_spec,
        sampler_spec=config.samplers,
        n_step=train_step)

    train_loader = DataLoader(
        dataset,
        batch_sampler=train_sampler,
        num_workers=config.get('loader_workers', cpu_count()),
        worker_init_fn=lambda w: np.random.seed(
            np.random.randint(2 ** 29) + w),
        collate_fn=collate_by_task,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    print(f"{Fore.GREEN}Creating Validation dataset{Style.RESET_ALL}")
    dataset_cfg.mode = 'val'
    val_dataset = instantiate(dataset_cfg)
    val_step = int(config.get('epochs') *
                   int(len(val_dataset)/config.get('vsize')))
    print(f"{Fore.GREEN}Number of val/step {val_step}{Style.RESET_ALL}")

    samplerClass = BatchSampler
    val_sampler = samplerClass(
        task_to_idx=val_dataset.task_to_idx,
        subtask_to_idx=val_dataset.subtask_to_idx,
        tasks_spec=dataset_cfg.tasks_spec,
        sampler_spec=config.samplers,
        n_step=val_step)
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=config.get('loader_workers', cpu_count()),
        worker_init_fn=lambda w: np.random.seed(
            np.random.randint(2 ** 29) + w),
        collate_fn=collate_by_task,
        pin_memory=False,
        prefetch_factor=2,
        persistent_workers=True
    )

    return train_loader, val_loader


def collect_stats(step, task_losses, task_accuracy, prefix='train'):
    """ create/append to stats dict of a one-layer dict structure:
        {'task_name/loss_key': [..], 'loss_key/task_name':[...]}"""
    tr_print = ''

    tr_print += f"{prefix} - Step {step}"
    for task_name, loss in task_losses.items():
        tr_print += f"\nTask {task_name}: "
        for loss_name,  loss_value in loss.items():
            tr_print += f"{loss_name}:  {round(loss_value.item(),2)} "
    for task_name, accuracy in task_accuracy.items():
        if 'global' not in task_name:
            tr_print += f"\nTask {task_name}: "
            for accuracy_name,  accuracy_value in accuracy.items():
                tr_print += f"{accuracy_name}:  {round(accuracy_value,2)} "

    return tr_print


def compute_accuracy(obj_logits, target_logits, obj_gt, target_gt):
    # Compute predictions by taking argmax along appropriate dimension
    obj_predictions = obj_logits.argmax(dim=1)
    target_predictions = target_logits.argmax(dim=1)

    obj_gt = obj_gt.argmax(dim=1)
    target_gt = target_gt.argmax(dim=1)

    # Compute accuracy for object classification
    obj_correct = (obj_predictions == obj_gt).sum().item()
    obj_total = obj_gt.size(0)
    obj_accuracy = obj_correct / obj_total

    # Compute accuracy for target classification
    target_correct = (target_predictions == target_gt).sum().item()
    target_total = target_gt.size(0)
    target_accuracy = target_correct / target_total

    return obj_accuracy, target_accuracy


def node_classification_loss(config, train_cfg, device, model, task_inputs, obj_loss, target_loss, val=False):
    model_inputs = defaultdict()
    task_to_idx = dict()
    task_losses = OrderedDict()
    task_accuracy = OrderedDict()
    start = 0

    for idx, (task_name, inputs) in enumerate(task_inputs.items()):
        for key in inputs.keys():
            if key != 'demo_data':
                model_inputs[key] = inputs[key].to(device)
            else:
                for key in inputs['demo_data'].keys():
                    model_inputs[key] = inputs['demo_data'][key].to(device)

        task_bsize = inputs['node_features'].shape[0]
        task_to_idx[task_name] = [start + i for i in range(task_bsize)]
        task_losses[task_name] = OrderedDict()
        task_accuracy[task_name] = OrderedDict()
        start += task_bsize

    all_losses = dict()
    all_accuracy = dict()

    out = model(
        inputs=model_inputs,
        inference=val
    )

    if config.lcgnet_conf.BUILD_NODE_CLASSIFIER:
        obj_logits = out[0].squeeze()
        target_logits = out[1].squeeze()
        # compute loss
        loss_obj = obj_loss(input=obj_logits,
                            target=model_inputs['obj_class'])
        loss_target = target_loss(
            input=target_logits,
            target=model_inputs['target_class'])
        complete_loss = loss_obj+loss_target

        # compute global accuracy
        obj_accuracy, target_accuracy = compute_accuracy(obj_logits=obj_logits,
                                                         target_logits=target_logits,
                                                         obj_gt=model_inputs['obj_class'],
                                                         target_gt=model_inputs['target_class'])

        all_losses['obj_loss'] = loss_obj
        all_losses['target_loss'] = loss_target
        all_losses['loss_sum'] = complete_loss
        task_accuracy['global_obj_accuracy'] = obj_accuracy
        task_accuracy['global_target_accuracy'] = target_accuracy

    for (task_name, idxs) in task_to_idx.items():
        for (loss_name, loss_val) in all_losses.items():
            if len(loss_val.shape) > 0:
                task_losses[task_name][loss_name] = torch.mean(loss_val[idxs])

    # compute accuracy for each task
    for (task_name, idxs) in task_to_idx.items():
        # compute accuracy
        obj_accuracy, target_accuracy = compute_accuracy(obj_logits=obj_logits[idxs],
                                                         target_logits=target_logits[idxs],
                                                         obj_gt=model_inputs['obj_class'][idxs],
                                                         target_gt=model_inputs['target_class'][idxs])
        task_accuracy[task_name]["obj_accuracy"] = obj_accuracy
        task_accuracy[task_name]["target_accuracy"] = target_accuracy

    return task_losses, task_accuracy
