import os
import json
import yaml
import copy
import torch
import hydra
import numpy as np
import torch.nn as nn
from os.path import join
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from collections import defaultdict
from hydra.utils import instantiate
# need for val. loader
from multi_task_il.datasets.sampler import DIYBatchSampler, TrajectoryBatchSampler
from multi_task_il.datasets.utils import collate_by_task 
from  multi_task_il.datasets.loss_functions import *
from omegaconf import OmegaConf
from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup
from multi_task_il.utils.lr_scheduler import build_scheduler, BaseScheduler
from multi_task_il.utils.early_stopping import EarlyStopping
import wandb
from torchsummary import summary
from tqdm import tqdm
import learn2learn as l2l
import gc
from colorama import Fore, Back
import torch.distributed as dist
import time



torch.autograd.set_detect_anomaly(True)
# for visualization
# MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 3, 1, 1))
# STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 3, 1, 1))
DEBUG = False

def loss_to_scalar(loss):
    x = loss.item()
    return float("{:.5f}".format(x))


def check_train_val_overlap(train_dataset, val_dataset):
    same_agent_file_cnt = 0
    same_demo_file_cnt = 0
    for task in train_dataset.agent_files.keys():
        for id in train_dataset.agent_files[task].keys():
            for agent_trj in train_dataset.agent_files[task][id]:
                if agent_trj in val_dataset.agent_files[task][id]:
                    same_agent_file_cnt += 1

    for task in train_dataset.demo_files.keys():
        for id in train_dataset.demo_files[task].keys():
            for demo_trj in train_dataset.demo_files[task][id]:
                if demo_trj in val_dataset.demo_files[task][id]:
                    same_demo_file_cnt += 1
    print(f"Overlapping counter {same_agent_file_cnt} - {same_demo_file_cnt}")


def worker_init_fn(worker_id):
    np.random.seed(np.random.randint(2 ** 29) + worker_id)

def make_model(config, local_rank):
   
    resume = config.get('resume', False)
    finetune = config.get('finetune', False)
    
    
    model = hydra.utils.instantiate(config.policy)
    model = model.to(torch.device("cuda:" + str(local_rank)))
    try:
        config.use_daml = 'DAMLNetwork' in config.policy._target_
        if config.use_daml:
            print("Switching to l2l.algorithms.MAML")
            model = l2l.algorithms.MAML(
                model,
                lr=config['policy']['maml_lr'],
                first_order=config['policy']['first_order'],
                allow_unused=True)
    except:
        print("use_daml not in configuration file")

    print("Model initialized to: {}".format(config.policy._target_))
    if resume or finetune:
        rpath = join(config.save_path, config.resume_path,
                            f"model_save-{config.resume_step}.pt")
        assert os.path.exists(rpath), "Can't seem to find {} anywhere".format(
            rpath)
        print('Finetuning model: load model from ...%s' %
                rpath)
        state_dict = torch.load(
            rpath, map_location=torch.device("cuda:" + str(local_rank)))
        if finetune:
            state_dict_keys = list(state_dict.keys())
            for key in state_dict_keys:
                if '_object_detector' in key or '_cond_backbone' in key or '_agent_backbone' in key:
                    state_dict.pop(key)
        model.load_state_dict(state_dict,strict=False)
        optimizer_state_dict = None
        if resume:
            try:
                # create path for loading state dict
                optimizer_state_dict = join(
                    config.save_path, config.resume_path, f"model_save-optim.pt")
                optimizer_state_dict = torch.load(
                    optimizer_state_dict, map_location=torch.device("cuda:" + str(local_rank)))
            except:
                print("Exception during loading optimizer state dict")
                optimizer_state_dict = None
    else:
        optimizer_state_dict = None
        
    return model, optimizer_state_dict

def make_optimizer_schedule( optimizer, optim_weights, optimizer_state_dict, config):
    
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            optim_weights,
            config.lr,
            weight_decay=config.get('weight_decay', 0))
    elif optimizer == 'RMSProp':
        optimizer = torch.optim.RMSprop(
            optim_weights,
            config.lr,
            weight_decay=config.get('weight_decay', 0))
    elif optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(
            optim_weights,
            config.lr,
            weight_decay=config.weight_decay)
        

    
    if optimizer_state_dict:
        optimizer.load_state_dict(optimizer_state_dict)

    print(
        f"Creating {optimizer}, with lr {optimizer.param_groups[0]['lr']}")

    lr_schedule = dict()
    if config.lr_schedule == 'None':
        lr_schedule['type'] = None
    else:
        lr_schedule['type'] = config.lr_schedule
    print(f"Lr-scheduler {config.lr_schedule}")
    
    return optimizer, build_scheduler(optimizer, lr_schedule)


def make_loss_function(config):
    
    loss_function = None
    if "VideoImitation" in config.policy._target_ or "InverseImitation" in config.policy._target_ or "DAMLNetwork" in config.policy._target_ or "CondPolicy" in config.policy._target_:
        if "grad_norm" not in config.get("loss", ""):
            loss_function = calculate_task_loss
        else:
            loss_function = calculate_grad_norm_loss

    elif "vima" in config.policy._target_:
        loss_function = loss_function_vima
    elif "cond_target_obj_detector" in config.policy._target_:
        loss_function = loss_func_bb

    return loss_function

def make_data_loaders(config, dataset_cfg, num_replicas: int = 1, global_rank: int = 1):
    """ Use .yaml cfg to create both train and val dataloaders """
    assert '_target_' in dataset_cfg.keys(), "Let's use hydra-config from now on. "
    print("Initializing {} with hydra config. \n".format(dataset_cfg._target_))
    print(
        f"---- Number of workder {config.get('loader_workers', cpu_count())}-----")
    dataset_cfg.mode = 'train'
    dataset = instantiate(dataset_cfg)
    train_step = int(config.get('epochs') *
                     int(len(dataset)/(num_replicas*config.get('bsize'))))
    epoch_step = int(len(dataset)/(num_replicas*config.get('bsize')))
    
    if not dataset_cfg.change_command_epoch:
        train_sampler = DIYBatchSampler(
            task_to_idx=dataset.task_to_idx,
            subtask_to_idx=dataset.subtask_to_idx,
            tasks_spec=dataset_cfg.tasks_spec,
            object_distribution_to_indx=dataset.object_distribution_to_indx,
            sampler_spec=config.samplers,
            n_step=train_step,
            dataset=dataset,
            num_replicas=num_replicas,
            rank=global_rank)
    else:
        train_sampler = TrajectoryBatchSampler(
            dataset,
            agent_files=dataset.all_agent_files,
            task_to_idx=dataset.task_to_idx,
            subtask_to_idx=dataset.subtask_to_idx,
            demo_task_to_idx=dataset.demo_task_to_idx,
            demo_subtask_to_idx=dataset.demo_subtask_to_idx,
            tasks_spec=dataset_cfg.tasks_spec,
            object_distribution_to_indx=dataset.object_distribution_to_indx,
            sampler_spec=config.samplers,
            n_step=train_step,
            epoch_steps=epoch_step,
            num_replicas=num_replicas,
            rank=global_rank,
        )

    train_loader = DataLoader(
        dataset,
        batch_sampler=train_sampler,
        num_workers=config.get('loader_workers', cpu_count()),
        collate_fn=collate_by_task,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    val_loader = None
    if dataset_cfg.split[1] > 0.0:
        dataset_cfg.mode = 'val'
        val_dataset = instantiate(dataset_cfg)
        # allow validation batch to have a different size
        config.samplers.batch_size = config.train_cfg.val_size
        val_step = int(config.get('epochs') *
                       int(len(val_dataset)/config.get('vsize')))

        if not dataset_cfg.change_command_epoch:
            val_sampler = DIYBatchSampler(
                task_to_idx=val_dataset.task_to_idx,
                subtask_to_idx=val_dataset.subtask_to_idx,
                tasks_spec=dataset_cfg.tasks_spec,
                object_distribution_to_indx=val_dataset.object_distribution_to_indx,
                sampler_spec=config.samplers,
                n_step=val_step,
                dataset = val_dataset,
                num_replicas=num_replicas,
                rank=global_rank)
        else:
            val_sampler = TrajectoryBatchSampler(
                val_dataset,
                agent_files=val_dataset.all_agent_files,
                task_to_idx=val_dataset.task_to_idx,
                subtask_to_idx=val_dataset.subtask_to_idx,
                demo_task_to_idx=val_dataset.demo_task_to_idx,
                demo_subtask_to_idx=val_dataset.demo_subtask_to_idx,
                tasks_spec=dataset_cfg.tasks_spec,
                object_distribution_to_indx=val_dataset.object_distribution_to_indx,
                sampler_spec=config.samplers,
                n_step=train_step,
                epoch_steps=int(len(val_dataset)/(num_replicas*config.get('bsize'))),
                num_replicas=num_replicas,
                rank=global_rank,
            )

        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            num_workers=config.get('loader_workers', cpu_count()),
            collate_fn=collate_by_task,
            pin_memory=False,
            prefetch_factor=2,
            persistent_workers=True
        )

    # check_train_val_overlap(train_dataset=dataset, val_dataset=val_dataset)
    return train_loader, val_loader


def collect_stats(step, task_losses, raw_stats, prefix='train'):
    """ create/append to stats dict of a one-layer dict structure:
        {'task_name/loss_key': [..], 'loss_key/task_name':[...]}"""
    task_names = sorted(task_losses.keys())
    for task, stats in task_losses.items():
        # expects: {'button': {"loss_sum": 1, "l_bc": 1}}
        for k, v in stats.items():
            for log_key in [f"{prefix}/{task}/{k}", f"{prefix}/{k}/{task}"]:
                if log_key not in raw_stats.keys():
                    raw_stats[log_key] = []
                raw_stats[log_key].append(loss_to_scalar(v))
        if "step" in raw_stats.keys():
            raw_stats["step"].append(int(step))
        else:
            raw_stats["step"] = [int(step)]
    tr_print = ""
    for i, task in enumerate(task_names):
        tr_print += "[{0:<9}] l_tot: {1:.4f} l_bc: {2:.4f} l_inv: {3: 4f} l_rep: {4: 4f} l_pnt: {5:.4f} l_aux: {6:.4f} avg_prec {7:.4f}".format(
            task,
            raw_stats.get(f"{prefix}/{task}/loss_sum", [0])[-1],
            raw_stats.get(f"{prefix}/{task}/l_bc", [0])[-1],
            raw_stats.get(f"{prefix}/{task}/l_inv", [0])[-1],
            raw_stats.get(f"{prefix}/{task}/rep_loss", [0])[-1],
            raw_stats.get(f"{prefix}/{task}/point_loss", [0])[-1],
            raw_stats.get(f"{prefix}/{task}/l_aux", [0])[-1],
            raw_stats.get(f"{prefix}/{task}/class_accuracy", [0])[-1],
        )
        if i % 3 == 2:  # use two lines to print
            tr_print += "\n"

    return tr_print


def generate_figure(images, context, fname='burner.png'):
    _B, T_im, _, _H, _W = images.shape
    T_con = context.shape[1]
    print("Images value range: ", images.min(), images.max(), context.max())
    print("Generating figures from images shape {}, context shape {} \n".format(
        images.shape, context.shape))
    npairs = 7
    skip = 8
    ncols = 4
    fig, axs = plt.subplots(nrows=npairs * 2, ncols=ncols, figsize=(
        ncols*3.5, npairs*2*2.8), subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(left=0.03, right=0.97, hspace=0.3, wspace=0.05)
    for img_index in range(npairs):
        show_img = images[img_index*skip].cpu().numpy()
        show_con = context[img_index*skip].cpu().numpy()
        for count in range(ncols):
            axs[img_index*2, count].imshow(show_img[count].transpose(1, 2, 0))
            if count < T_con:
                axs[img_index*2+1,
                    count].imshow(show_con[count].transpose(1, 2, 0))

    plt.tight_layout()
    print("Saving figure to: ", fname)
    plt.savefig(fname)


class Trainer:

    def __init__(self, allow_val_grad=False, hydra_cfg=None):
        assert hydra_cfg is not None, "Need to start with hydra-enabled yaml file!"
        
        self.config = hydra_cfg
        self.train_cfg = hydra_cfg.train_cfg
        # initialize device
        self._device_list = None
        self._device_list = self.device_list()
        print(f"List of devices {self._device_list}") 
        
        self._allow_val_grad = allow_val_grad
        self._world_size = hydra_cfg.get('num_gpus', 1) * hydra_cfg.get('num_nodes', 1)

        self.task_names = [task['name'] for task in self.config.tasks]
    
        # set of file saving
        if not os.path.exists(self.config.save_path):
            os.makedirs(self.config.save_path)

        assert self.config.exp_name != -1, 'Specify an experiment name for log data!'
        self._best_validation_loss = float('inf')
        self._best_validation_weights = None

        append = "-Batch{}".format(int(self.config.bsize))
        if 'mosaic' in hydra_cfg.policy:
            append = "-Batch{}-{}gpu-Attn{}ly{}-Act{}ly{}mix{}".format(
                int(self.config.bsize), int(torch.cuda.device_count()),
                int(self.config.policy.attn_cfg.n_attn_layers), int(
                    self.config.policy.attn_cfg.attn_ff),
                int(self.config.policy.action_cfg.n_layers), int(
                    self.config.policy.action_cfg.out_dim),
                int(self.config.policy.action_cfg.n_mixtures))

            if self.config.policy.concat_demo_head:
                append += "-headCat"
            elif self.config.policy.concat_demo_act:
                append += "-actCat"
            else:
                append += "-noCat"
            if 'mosaic' in hydra_cfg.policy:
                append += "-simclr{}x{}".format(int(self.config.policy.simclr_config.compressor_dim), int(
                    self.config.policy.simclr_config.hidden_dim))

        self.config.exp_name += append

        save_dir = join(self.config.get('save_path', './'),
                        str(self.config.exp_name))
        save_dir = os.path.expanduser(save_dir)
        self._save_fname = join(save_dir, 'model_save')
        self.save_dir = save_dir
        print(f"Saving dir {self.save_dir}")
        self._step = 0
        self._epoch = 0

        # create early stopping object
        self._early_stopping = EarlyStopping(patience=self.train_cfg.early_stopping.patience,
                                            verbose=True,
                                            delta=self.train_cfg.early_stopping.delta,
                                            path=self.save_dir
                                            )

    def worker(self, local_rank, *args):
        
        global_rank = args[0] * args[1] + local_rank 
        dist.init_process_group( 
        backend='nccl',  
        world_size=self._world_size, 
        rank=global_rank 
        )
        
        if global_rank == 0:
                        
            if self.config.wandb_log:
                config_keys = ['train_cfg', 'tasks', 'samplers', 'dataset_cfg', 'policy']
                # for k in config_keys:
                #     print(k, self.config.get(k))
                #     print(k, dict(self.config.get(k)))
                #     print('-'*20)
                wandb_config = {k: self.config.get(k) for k in config_keys}
                wandb.login(key='227ed2fded06f63748a7a29dae55acdda7d131ff', relogin=True)
                print(f"Exp name: {self.config.exp_name}")
                self.config.project_name = self.config.exp_name.split('-Batch')[0]
                run = wandb.init(project=self.config.project_name,
                                name=self.config.exp_name,
                                config=wandb_config,
                                sync_tensorboard=False)
        
        
        self.train(num_replicas=self._world_size,
                   global_rank=global_rank,
                   local_rank=local_rank)
         

    def train_loop(self, train_loader, scheduler, loss_function, global_rank, local_rank, model, optimizer, task_loss_muls, raw_stats: dict = dict(), epoch: int = 0, log_freq: int = -1, print_freq: int = -1, frac: float = 0.0):
        
        #### ---- Train loop ----####
        model = model.train()
        
    
        if hasattr(model.module, '_object_detector') :
            print(f"Object detector is set to eval mode")
            if model.module._object_detector is not None: 
                model.module._object_detector.eval()
                print(f"Object detector mode {model.module._object_detector.training}")    
            
        train_step = len(train_loader)
        print(f"Training for {train_step} steps")
        epoch_steps = 0
        for inputs in tqdm(train_loader):
            torch.cuda.empty_cache()
            
            # calculate loss here:
            # if global_rank == 0:
            #     start_inference = time.time()
            task_losses = loss_function(
                self.config, self.train_cfg, self._device_list[local_rank], model, inputs)
            # if global_rank == 0:
            #     end_inference = time.time()
            #     print("Inference time: ", end_inference - start_inference)
            
            if "grad_norm" not in self.config.get("loss", ""):
                optimizer.zero_grad()
                weighted_task_loss = sum(
                    [l["loss_sum"] * task_loss_muls.get(name) for name, l in task_losses.items()])
                weighted_task_loss.backward()
                optimizer.step()
            else:
                raise Exception("Grad Norm not implemented yet")        

            if getattr(model, '_load_contrastive', False) and not 'cond_target_obj_detector' in self.config.policy._target_:
                # update target params
                mod = model.module if isinstance(model, nn.DataParallel) else model
                if self.train_cfg.target_update_freq > -1:
                    mod.momentum_update(frac)
                    if self._step % self.train_cfg.target_update_freq == 0:
                        mod.soft_param_update()    
                
            # log stats
            # calculate train iter stats
            if global_rank == 0:
                tolog = dict()
                
                if self._step % log_freq == 0:
                    train_print = collect_stats(
                        self._step, task_losses, raw_stats, prefix='train')
                    
                    if self.config.wandb_log:
                        tolog['train_step'] = self._step
                        tolog['epoch'] = epoch
                        i = 0
                        for task_name, losses in task_losses.items():
                            if "grad_norm" in self.config.get("loss", ""):
                                #tolog[f'train/weight_loss_{task_name}'] = weights_loss[i]
                                raise Exception("Grad Norm not implemented yet")  
                            for loss_name, loss_val in losses.items():
                                tolog[f'train/{loss_name}/{task_name}'] = loss_val
                                tolog[f'train/{task_name}/{loss_name}'] = loss_val
                            i += 1

                    if self._step % print_freq == 0:
                        epoch_steps += 1
                        print(f"Epoch perc {epoch_steps / train_step}")
                        print(
                            'Training epoch {1}/{2}, step {0}: \t '.format(self._step, epoch, self.config.epochs))
                        print(train_print)
                        
                    if scheduler != 'None' and self.config.cosine_annealing:
                        if self.config.wandb_log:
                            # log learning-rate
                            tolog['learning_rate'] = scheduler.optimizer.param_groups[0]['lr']
                            
                    if self.config.wandb_log:
                        wandb.log(tolog)
                    
                self._step += 1


    def val_loop(self, val_loader, scheduler, loss_function, global_rank, local_rank, model, optimizer, task_loss_muls, task_names, raw_stats: dict = dict(), epoch: int = 0,):
             
        validate = True
        tolog = dict()
        model = model.eval()
        
        if "CondTargetObjectDetector" in self.config.policy._target_:
            if not self.config.get("use_daml", False) and val_loader is not None:
                validate = True
        else:
            if (((epoch % 10 == 0) or (epoch == self.config.epochs-1)) and not self.config.get("use_daml", False)) and val_loader is not None:
                validate= True
        
        if validate:
            print("Validation")
            rollout = self.config.get("rollout", False)
           
            
            if not rollout:
                
                # exhaust all data in val loader and take avg loss
                all_val_losses = {task: defaultdict(
                    list) for task in task_names}
                
                # val_iter = iter(val_loader)
                # for i, val_inputs in tqdm(enumerate(val_loader), total=len(val_loader)):
                for i, val_inputs in tqdm(enumerate(val_loader), total=len(val_loader)):
                    use_daml = self.config.get("use_daml", False)
                    if use_daml:  # allow grad!
                        val_task_losses = loss_function(
                                            self.config, 
                                            self.train_cfg, 
                                            self._device_list[local_rank], 
                                            model, 
                                            val_inputs)
                    else:
                        with torch.no_grad():
                            val_task_losses = loss_function(
                                self.config,             
                                self.train_cfg, 
                                self._device_list[local_rank], 
                                model, 
                                val_inputs,
                                val=False)

                    for task, losses in val_task_losses.items():
                        for k, v in losses.items():
                            all_val_losses[task][k].append(v)

                # take average across all batches in the val loader
                avg_losses = dict()
                for task, losses in all_val_losses.items():
                    avg_losses[task] = {
                        k: torch.mean(torch.stack(v)) for k, v in losses.items()}

                # compute the sum of validation losses
                weighted_task_loss_val = sum(
                    [l["loss_sum"] * task_loss_muls.get(name) for name, l in avg_losses.items()])
                
                if global_rank == 0:
                    val_print = collect_stats(
                    self._step, avg_losses, raw_stats, prefix='val')
                    
                    print('Validation step {}:'.format(self._step))
                    print(val_print)

                    if self.config.wandb_log:
                        # log learning-rate
                        tolog['validation_step'] = self._step
                        tolog['validation_epoch'] = epoch
                        for task_name, losses in avg_losses.items():
                            for loss_name, loss_val in losses.items():
                                tolog[f'val/{loss_name}/{task_name}'] = loss_val
                                tolog[f'val/{task_name}/{loss_name}'] = loss_val
                                
                        if not(isinstance(scheduler, BaseScheduler)):
                            tolog['learning_rate'] = scheduler._schedule.optimizer.param_groups[0]['lr']
                        wandb.log(tolog)
                                    
                return weighted_task_loss_val
            
            elif rollout:
                raise Exception("Rollout not implemented yet")
            
                # elif rollout:  # and self._step != 0:
                #     import functools
                #     from torch.multiprocessing import Pool
                #     from train_any import seed_everything
                #     from multi_task_test.test_any_task import _proc
                #     del inputs
                #     gc.collect()
                #     torch.cuda.empty_cache()

                #     target_obj_dec = None
                #     controller_path = "/home/rsofnc000/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json"
                #     model_name = self.config.policy._target_

                #     for task in self.tasks:
                #         import random
                #         task_name = task['name']
                #         results_dir = os.path.join(
                #             self.save_dir, 'results_{}_{}/'.format(task_name, e))
                #         os.makedirs(results_dir, exist_ok=True)
                #         seed_everything(seed=42)
                #         n_run_per_task = 5
                #         N_step = 95
                #         if "MOSAIC" in self.config.exp_name or "Double" in self.config.exp_name:
                #             gt_bb = True if model._concat_bb and model._object_detector is None else False
                #         else:
                #             gt_bb = False
                #         place = True if 'Double' in self.config.exp_name else False
                #         task_cfg = self.config["tasks_cfgs"][task['name']] if 'button' not in task_name else self.config["tasks_cfgs"]['button']
                #         if len(task_cfg.get('skip_ids', [])) == 0:
                #             n_run = int(n_run_per_task *
                #                         task_cfg.get('n_tasks', 0))
                #             variation = None
                #         else:
                #             n_run = int(n_run_per_task *
                #                         len(task_cfg.get('skip_ids', [])))
                #             variation = task_cfg.get(
                #                 'skip_ids', [])
                #         if 'button' in task_name:
                #             task_name_proc = 'button'
                #         else:
                #             task_name_proc = task_name
                #         f = functools.partial(_proc,
                #                                 model,
                #                                 self.config,
                #                                 results_dir,
                #                                 200,
                #                                 360,
                #                                 False,
                #                                 False,
                #                                 False,
                #                                 task_name_proc,
                #                                 None,
                #                                 variation,
                #                                 N_step,
                #                                 controller_path,
                #                                 model_name,
                #                                 local_rank,
                #                                 False,
                #                                 gt_bb,
                #                                 -1,
                #                                 -1,
                #                                 False,
                #                                 place
                #                                 )
                #         seeds = [(random.getrandbits(32), i, None)
                #                     for i in range(n_run)]
                #         with Pool(10) as p:
                #             task_success_flags = p.starmap(f, seeds)
                #         if "CondTargetObjectDetector" in self.config.policy._target_:
                #             to_log = dict()
                #             all_mean_iou = [t['avg_iou']
                #                             for t in task_success_flags]
                #             all_fp = [t['avg_fp']
                #                         for t in task_success_flags]
                #             all_tp = [t['avg_tp']
                #                         for t in task_success_flags]
                #             to_log['avg_iou'] = np.mean(all_mean_iou)
                #             to_log['avg_fp'] = np.mean(all_fp)
                #             to_log['avg_tp'] = np.mean(all_tp)
                #             if to_log['avg_tp'] >= best_tp:
                #                 print(
                #                     f"Saving best model, from {best_tp} to {to_log['avg_tp']}")
                #                 best_tp = to_log['avg_tp']
                #                 self.save_checkpoint(
                #                     model, optimizer, weights_fn, save_fn, to_log['avg_tp'])
                #         else:
                #             to_log = dict()
                #             flags = dict()
                #             for t in task_success_flags:
                #                 for k in t.keys():
                #                     if flags.get(k, None) is None:
                #                         flags[k] = [int(t[k])]
                #                     else:
                #                         flags[k].append(int(t[k]))
                #             for k in flags.keys():
                #                 avg_flags = np.mean(flags[k])
                #                 to_log[f'avg_{k}'] = avg_flags
                #             to_log[f'epoch'] = e
                #             wandb.log(to_log)

                #             # if tolog['avg_success'] >= best_avg_success:
                #             # print(
                #             #     f"Save model best_avg_success from {best_avg_success} to {tolog['avg_success']}")
                #             # best_avg_success = tolog['avg_success']
                #             self.save_checkpoint(
                #                 model, optimizer, weights_fn, save_fn, str(round(to_log['avg_success'], 3)).replace('.','_'))

                #         if self.config.wandb_log:
                #             wandb.log(to_log)


    def train(self, weights_fn=None, save_fn=None, optim_weights=None, num_replicas: int = 1, global_rank: int = 0, local_rank: int = 0):

        model, optimizer_state_dict = make_model(self.config, local_rank)
        
        optim_weights = optim_weights if optim_weights is not None else model.parameters()
        optimizer, lr_scheduler = make_optimizer_schedule(self.config.train_cfg.optimizer,
                                                          optim_weights,
                                                          optimizer_state_dict,
                                                          self.config.train_cfg)
        
        # GradNorm flag
        tasks = self.config.tasks
        if "grad_norm" in self.config.get("loss", ""):
            dict_task_name_weight_indx = dict()
            for indx, task in enumerate(tasks):
                dict_task_name_weight_indx[task['name']] = indx
        else:
            # grad_norm is not requested
            sum_mul = sum([task.get('loss_mul', 1) for task in tasks])
            task_loss_muls = {task.name:
                                float("{:3f}".format(task.get('loss_mul', 1) / sum_mul)) for task in tasks}
            print(" Weighting each task loss separately:", task_loss_muls)
        
        
        loss_function = make_loss_function(self.config) 

        train_loader, val_loader = make_data_loaders(
            self.config, 
            self.config.train_cfg.dataset,
            num_replicas=num_replicas,
            global_rank=global_rank)
        
        dist.barrier()
        
        # wrap model in DataParallel if needed and transfer to correct device
        print('\n-------------------\nTraining stage\nFound {} GPU devices \n'.format(self.device_count))

        
        print('Model on device: {}'.format("cuda:" + str(local_rank) if torch.cuda.is_available() else "cpu"))
        device = torch.device("cuda:" + str(local_rank) if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model = nn.parallel.DistributedDataParallel(model, 
                                                    device_ids=[local_rank],
                                                    find_unused_parameters=True)
        dist.barrier()

        # initialize constants:
        # compute epochs
        if self.config.resume:
            # epochs = self.config.epochs - \
            #     int(self.config.resume_step/len(train_loader))
            # print(f"\n---- Remaining epochs {epochs} ----\n")
            # self._step = int(self.config.resume_step)
            # print(f"\n----Starting step {self._step} ----\n")
            
            # remaining epochs
            epochs = self.config.epochs #- (self.config.resume_step + 1)
            self._step = len(train_loader) * (self.config.resume_step +1)
            print(f"\n---- Remaining epochs {self.config.epochs - (self.config.resume_step+1)} ----\n")
            print(f"\n----Starting step {self._step} ----\n")
            
        else:
            epochs = self.train_cfg.get('epochs', 1)
            self._step = 0
            self.config.resume_step = 0

        vlm_alpha = self.train_cfg.get('vlm_alpha', 0.6)
        log_freq = self.train_cfg.get('log_freq', 1000)
        val_freq = self.train_cfg.get('val_freq', 1000)
        print_freq = self.train_cfg.get('print_freq', 10000)
        save_freq = self.train_cfg.get('save_freq', 10000)
        if save_freq == -1:
            save_freq = len(train_loader)
        if val_freq == -1:
            val_freq = len(train_loader)
        print(f"Save frequency {save_freq}")
        print(f"Val frequency {val_freq}")

        try:
            print("\n----Loss multipliers: \n BC: {} inv: {} Point: {}\n----".format(
                self.train_cfg.bc_loss_mult, self.train_cfg.inv_loss_mult, self.train_cfg.pnt_loss_mult))

            print(
                {name: mul for name, mul in self.train_cfg.rep_loss_muls.items() if mul != 0})
            if self.train_cfg.bc_loss_mult == 0 and self.train_cfg.inv_loss_mult == 0:
                assert sum([v for k, v in self.train_cfg.rep_loss_muls.items()]
                           ) != 0, self.train_cfg.rep_loss_muls
        except:
            pass

        self.generated_png = False
        
        if val_loader != None:
            # val_iter = iter(val_loader)
            print(f"Training for {epochs} epochs train dataloader has length {len(train_loader)}, \ which sums to {epochs * len(train_loader)} total train steps, \ validation loader has length {len(val_loader)}")
        else:
            print(
                f"Training for {epochs} epochs train dataloader has length {len(train_loader)}")

        best_tp = 0
        best_avg_success = 0.0
        # # take the parameters of action modules
        # torch.stack((model._action_module.parameters(
        # ), model._inv_model.parameters()model._action_dist.parameters())))
        try:
            model_parameters = list()
            model_parameters.extend(list(model._action_module.parameters()))
            model_parameters.extend(list(model._inv_model.parameters()))
            model_parameters.extend(list(model._action_dist.parameters()))
        except:
            pass
        
        alpha = 0.16
        raw_stats = dict()
        dist.barrier()
        
        
        # save model
        # if global_rank == 0:
        #     self.save_checkpoint(model, optimizer, weights_fn, save_fn)
        
        for e in range(self.config.resume_step+1, epochs):
            self._epoch = e
            frac = e / epochs
            print(f"Training frac {frac}")
            # with tqdm(train_loader, unit="batch") as tepoch:
            
            self.train_loop(train_loader=train_loader,
                            scheduler=lr_scheduler,
                            loss_function=loss_function,
                            global_rank=global_rank,
                            local_rank=local_rank,
                            model=model,
                            optimizer=optimizer,
                            task_loss_muls=task_loss_muls,
                            raw_stats=raw_stats,
                            epoch=e,
                            log_freq=log_freq,
                            print_freq=print_freq,
                            frac=frac)
                
            dist.barrier()
            
            #### ---- Validation step ----####
            if val_loader is not None:
                val_metric = self.val_loop(val_loader=val_loader,
                                        scheduler=lr_scheduler, 
                                        loss_function=loss_function, 
                                        global_rank=global_rank, 
                                        local_rank=local_rank, 
                                        model=model, 
                                        optimizer=optimizer, 
                                        task_loss_muls=task_loss_muls, 
                                        task_names=self.task_names, 
                                        raw_stats=raw_stats, 
                                        epoch= e,)
                
                if not isinstance(val_metric, torch.Tensor):
                    val_metric = torch.tensor(val_metric)
                
                torch.distributed.all_reduce(val_metric, op=dist.ReduceOp.AVG)
                
                if global_rank == 0:
                    print(f"Val metric {val_metric}")
                
                
                if self.config.train_cfg.lr_schedule != 'None':
                    # perform lr-scheduling step
                    lr_scheduler.step(val_loss=val_metric)
                    
                
                # check for early stopping
                if self.train_cfg.early_stopping.patience != -1:
                    self._early_stopping(val_metric, 
                                        model, 
                                        self._epoch, 
                                        optimizer,
                                        rank=global_rank)
            dist.barrier()
            
            # save model
            if global_rank == 0:
                self.save_checkpoint(model, optimizer, weights_fn, save_fn)
            
            if self._early_stopping.early_stop:
                print('Early stopping after epoch {}'.format(e + 1))
                break            
            
                                

    def save_checkpoint(self, model, optimizer, weights_fn=None, save_fn=None, save_name=None):
        

        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

        if save_name is not None:
            torch.save(model_to_save.state_dict(),
                    self._save_fname +'-{}-{}.pt'.format(save_name,self._epoch))
        else:
            torch.save(model_to_save.state_dict(),
                    self._save_fname + '-{}.pt'.format(self._epoch))
            
        if self.config.get('save_optim', False):
            torch.save(optimizer.state_dict(), self._save_fname +
                       '-optim.pt')
        
        print(f'Model checkpoint saved at epoch {self._epoch}')
        return

    @property
    def device_count(self):
        return torch.cuda.device_count()

    def device_list(self):
        if self._device_list is None:
            dev_list = []
            for i in range(torch.cuda.device_count()):
                print(f"Adding device {i}")
                dev_list.append(torch.device("cuda:{}".format(i)))
            return dev_list
        return copy.deepcopy(self._device_list)


    def _loss_to_scalar(self, loss):
        """For more readable logging"""
        x = loss.item()
        return float("{:.3f}".format(x))

    @property
    def step(self):
        if self._step is None:
            raise Exception("Optimization has not begun!")
        return self._step

    @property
    def is_img_log_step(self):
        return self._step % self._img_log_freq == 0


class Workspace(object):
    """ Initializes the policy model and prepare for Trainer.train() """

    def __init__(self, cfg):
        
        self.trainer = Trainer(allow_val_grad=False, hydra_cfg=cfg)
        print("Finished initializing trainer")
        self.config = self.trainer.config
        
        # map between task and number of tasks
        n_tasks = []
        tasks = dict()
        start = 0
        for i, task in enumerate(cfg.tasks):
            n_tasks.append(task['n_tasks'])
            tasks[task['name']] = (start, task['n_tasks'])
            start += task['n_tasks']

        # move log path to here!
        print('\n----Done initializing Workspace, saving config.yaml to directory: {}----\n'.format(
            self.trainer.save_dir))

        try:
            os.makedirs(self.trainer.save_dir, exist_ok=(
                'burn' in self.trainer.save_dir))
            os.makedirs(join(self.trainer.save_dir, 'stats'), exist_ok=True)
        except:
            pass

        save_config = copy.deepcopy(self.trainer.config)
        OmegaConf.save(config=save_config, f=join(
            self.trainer.save_dir, 'config.yaml'))

    def run(self):

        torch.multiprocessing.spawn(self.trainer.worker,
                                    nprocs=self.config.num_gpus, 
                                    args=(  self.config.node_id,
                                            self.config.num_gpus,
                                            self.config))
        
        print("Done training")
