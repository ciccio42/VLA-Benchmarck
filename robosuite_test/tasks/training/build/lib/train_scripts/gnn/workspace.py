import hydra
import torch
import os
import copy
from omegaconf import OmegaConf
from os.path import join
import wandb
from multi_task_il.utils.lr_scheduler import build_scheduler
from multi_task_il.utils.early_stopping import EarlyStopping
import torch.nn as nn
from utils import make_data_loaders, node_classification_loss, collect_stats
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style

colorama_init()


class Trainer:

    def __init__(self, allow_val_grad=False, hydra_cfg=None):
        assert hydra_cfg is not None, "Need to start with hydra-enabled yaml file!"
        self.config = hydra_cfg
        self.train_cfg = hydra_cfg.train_cfg
        # initialize device

        def_device = hydra_cfg.device if hydra_cfg.device != -1 else 0
        self._device_id = def_device
        self._device_list = None
        self._device = None
        try:
            self._device = torch.device("cuda:{}".format(def_device))
            self._allow_val_grad = allow_val_grad
        except:
            self._device = torch.device("cuda:{}".format(def_device[0]))
            self._device_list = self.device_list()
        # set of file saving

        if not os.path.exists(self.config.save_path):
            os.makedirs(self.config.save_path)

        assert self.config.exp_name != -1, 'Specify an experiment name for log data!'
        self._best_validation_loss = float('inf')
        self._best_validation_weights = None

        append = "-Batch{}".format(int(self.config.bsize))

        self.config.exp_name += append

        save_dir = join(self.config.get('save_path', './'),
                        str(self.config.exp_name))
        save_dir = os.path.expanduser(save_dir)
        self._save_fname = join(save_dir, 'model_save')
        self.save_dir = save_dir
        print(f"{Fore.GREEN}Saving dir {self.save_dir}{Style.RESET_ALL}")
        self._step = None
        if self.config.wandb_log:
            config_keys = ['train_cfg', 'tasks',
                           'samplers', 'dataset_cfg', 'policy']
            # for k in config_keys:
            #     print(k, self.config.get(k))
            #     print(k, dict(self.config.get(k)))
            #     print('-'*20)
            wandb_config = {k: self.config.get(k) for k in config_keys}
            run = wandb.init(project=self.config.project_name,
                             name=self.config.exp_name, config=wandb_config, sync_tensorboard=False)

        # create early stopping object
        self._early_stopping = EarlyStopping(patience=self.train_cfg.early_stopping.patience,
                                             verbose=True,
                                             delta=self.train_cfg.early_stopping.delta,
                                             path=self.save_dir
                                             )

    def train(self, model, weights_fn=None, save_fn=None, optim_weights=None, optimizer_state_dict=None, loss_function=None):

        self._train_loader, self._val_loader = make_data_loaders(
            self.config, self.train_cfg.dataset)
        # wrap model in DataParallel if needed and transfer to correct device
        print(
            f'{Fore.BLUE}\n-------------------\nTraining stage\nFound {self.device_count} GPU devices \n{Style.RESET_ALL}')

        # compute epochs
        if self.config.resume:
            epochs = self.config.epochs - \
                int(self.config.resume_step/len(self._train_loader))
            print(f"\n---- Remaining epochs {epochs} ----\n")
            self._step = int(self.config.resume_step)
            print(f"\n----Starting step {self._step} ----\n")
        else:
            epochs = self.train_cfg.get('epochs', 1)
            self._step = 0

        vlm_alpha = self.train_cfg.get('vlm_alpha', 0.6)
        log_freq = self.train_cfg.get('log_freq', 1000)
        val_freq = self.train_cfg.get('val_freq', 1000)
        print_freq = self.train_cfg.get('print_freq', 10000)
        save_freq = self.train_cfg.get('save_freq', 10000)
        if save_freq == -1:
            save_freq = len(self._train_loader)
        if val_freq == -1:
            val_freq = len(self._train_loader)
        print(f"Save frequency {save_freq}")
        print(f"Val frequency {val_freq}")

        raw_stats = dict()
        if self._val_loader != None:
            val_iter = iter(self._val_loader)
            print(f"Training for {epochs} epochs train dataloader has length {len(self._train_loader)}, \ which sums to {epochs * len(self._train_loader)} total train steps, \ validation loader has length {len(self._val_loader)}")
        else:
            print(
                f"Training for {epochs} epochs train dataloader has length {len(self._train_loader)}")

        step = 0
        best_fp = np.inf
        best_avg_success = 0.0

        alpha = 0.16

        if self.device_count > 1 and not isinstance(model, nn.DataParallel):
            print("Training stage \n Device list: {}".format(self._device_list))
            model = nn.DataParallel(model, device_ids=self._device_list).cuda()
        else:
            model = model.to(self._device)

        # initialize optimizer and lr scheduler
        optim_weights = optim_weights if optim_weights is not None else model.parameters()
        optimizer, scheduler = self._build_optimizer_and_scheduler(
            self.config.train_cfg.optimizer, optim_weights, optimizer_state_dict, self.train_cfg)

        model = model.train()
        if not isinstance(model, nn.DataParallel):
            model = model.to(self._device)
        # summary(model)
        obj_loss = nn.CrossEntropyLoss(reduction='none')
        target_loss = nn.CrossEntropyLoss(reduction='none')
        obj_loss_val = nn.CrossEntropyLoss(reduction='none')
        target_loss_val = nn.CrossEntropyLoss(reduction='none')
        for e in range(epochs):
            frac = e / epochs
            # with tqdm(self._train_loader, unit="batch") as tepoch:
            self.save_checkpoint(
                model, optimizer, weights_fn, save_fn)
            for inputs in tqdm(self._train_loader):
                tolog = {}
                # calculate loss here:
                task_losses, task_accuracy = loss_function(
                    self.config, self.train_cfg, self._device, model, inputs, obj_loss, target_loss)
                task_names = sorted(task_losses.keys())
                torch.cuda.empty_cache()
                # sum losses over tasks
                optimizer.zero_grad()
                weighted_task_loss = sum(
                    [l["loss_sum"] * 1.0 for name, l in task_losses.items()])
                weighted_task_loss.backward()
                optimizer.step()

                if self._step % log_freq == 0:
                    train_print = collect_stats(
                        self._step, task_losses, task_accuracy, prefix='train')

                    if self.config.wandb_log:
                        tolog['Train Step'] = self._step
                        tolog['Epoch'] = e
                        for task_name, losses in task_losses.items():
                            for loss_name, loss_val in losses.items():
                                tolog[f'train/{loss_name}/{task_name}'] = loss_val.item()
                                tolog[f'train/{task_name}/{loss_name}'] = loss_val.item()

                        for task_name, accuracy in task_accuracy.items():
                            if 'global' not in task_name:
                                for accuracy_name, accuracy_val in accuracy.items():
                                    tolog[f'train/{accuracy_name}/{task_name}'] = accuracy_val
                                    tolog[f'train/{task_name}/{accuracy_name}'] = accuracy_val

                    if self._step % print_freq == 0:
                        print(
                            'Training epoch {1}/{2}, step {0}: \t '.format(self._step, e, epochs))
                        print(train_print)

                if self.config.wandb_log:
                    wandb.log(tolog)

                if self._step % val_freq == 0:
                    # exhaust all data in val loader and take avg loss
                    all_val_losses = {task: defaultdict(
                        list) for task in task_names}
                    all_val_acc = {task: defaultdict(
                        list) for task in task_names}
                    model = model.eval()
                    for val_inputs in tqdm(self._val_loader):
                        with torch.no_grad():
                            val_task_losses, val_task_accuracy = loss_function(
                                self.config,
                                self.train_cfg,
                                self._device,
                                model,
                                val_inputs,
                                obj_loss_val,
                                target_loss_val,
                                val=True
                            )

                        for task, losses in val_task_losses.items():
                            for k, v in losses.items():
                                all_val_losses[task][k].append(v)

                        for task, accuracy in val_task_accuracy.items():
                            if 'global' not in task:
                                for k, v in accuracy.items():
                                    all_val_acc[task][k].append(v)

                    avg_losses = dict()
                    for task, losses in all_val_losses.items():
                        avg_losses[task] = {
                            k: torch.mean(torch.stack(v)) for k, v in losses.items()}

                    avg_accuracy = dict()
                    for task, accuracy in all_val_acc.items():
                        avg_accuracy[task] = dict()
                        if 'global' not in task:
                            for k, v in accuracy.items():
                                avg_accuracy[task][k] = np.mean(np.array(v))

                    val_print = collect_stats(
                        self._step, avg_losses, avg_accuracy, prefix='val')
                    print(val_print)
                    if self.config.wandb_log:
                        tolog['Val Step'] = self._step
                        tolog['Epoch'] = e
                        for task_name, losses in avg_losses.items():
                            for loss_name, loss_val in losses.items():
                                tolog[f'val/{loss_name}/{task_name}'] = loss_val.item()
                                tolog[f'val/{task_name}/{loss_name}'] = loss_val.item()

                        for task_name, accuracy in avg_accuracy.items():
                            if 'global' not in task_name:
                                for accuracy_name, accuracy_val in accuracy.items():
                                    tolog[f'val/{accuracy_name}/{task_name}'] = accuracy_val
                                    tolog[f'val/{task_name}/{accuracy_name}'] = accuracy_val

                        wandb.log(tolog)

                    model = model.train()
                self._step += 1

    def save_checkpoint(self, model, optimizer, weights_fn=None, save_fn=None):
        if save_fn is not None:
            save_fn(self._save_fname, self._step)
        else:
            save_module = model
            if weights_fn is not None:
                save_module = weights_fn()
            elif isinstance(model, nn.DataParallel):
                save_module = model.module
            torch.save(save_module.state_dict(),
                       self._save_fname + '-{}.pt'.format(self._step))
        if self.config.get('save_optim', False):
            torch.save(optimizer.state_dict(), self._save_fname +
                       '-optim.pt')
        print(f'Model checkpoint saved at step {self._step}')
        return

    @property
    def device_count(self):
        if self._device_list is None:
            return torch.cuda.device_count()
        return len(self._device_list)

    def device_list(self):
        if self._device_list is None:
            dev_list = []
            for i in range(torch.cuda.device_count()):
                dev_list.append(i)
            return dev_list
        return copy.deepcopy(self._device_list)

    @property
    def device(self):
        return copy.deepcopy(self._device)

    def _build_optimizer_and_scheduler(self, optimizer, optim_weights, optimizer_state_dict, cfg):
        assert self.device_list is not None, str(self.device_list)
        if optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                optim_weights,
                cfg.lr,
                weight_decay=cfg.get('weight_decay', 0))
        elif optimizer == 'RMSProp':
            optimizer = torch.optim.RMSprop(
                optim_weights,
                cfg.lr,
                weight_decay=cfg.get('weight_decay', 0))
        elif optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(
                optim_weights,
                cfg.lr,
                weight_decay=cfg.weight_decay)
        if optimizer_state_dict:
            optimizer.load_state_dict(optimizer_state_dict)

        print(
            f"Creating {optimizer}, with lr {optimizer.param_groups[0]['lr']}")

        lr_schedule = dict()
        if cfg.lr_schedule == 'None':
            lr_schedule['type'] = None
        else:
            lr_schedule['type'] = cfg.lr_schedule
        print(f"Lr-scheduler {cfg.lr_schedule}")
        return optimizer, build_scheduler(optimizer, lr_schedule)

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
        print(f"{Fore.YELLOW}Finished initializing trainer{Style.RESET_ALL}")
        config = self.trainer.config

        # map between task and number of tasks
        n_tasks = []
        tasks = dict()
        start = 0
        for i, task in enumerate(cfg.tasks):
            n_tasks.append(task['n_tasks'])
            tasks[task['name']] = (start, task['n_tasks'])
            start += task['n_tasks']

        resume = config.get('resume', False)
        finetune = config.get('finetune', False)

        # config.policy.n_tasks = n_tasks
        # config.dataset_cfg.tasks = tasks
        # config.dataset_cfg.n_tasks = int(np.sum(n_tasks))
        self.action_model = hydra.utils.instantiate(config.policy)

        print(
            f"{Fore.YELLOW}Model initialized to: {config.policy._target_}{Style.RESET_ALL}")
        if resume or finetune:
            self._rpath = os.path.join(cfg.save_path, cfg.resume_path,
                                       f"model_save-{cfg.resume_step}.pt")
            assert os.path.exists(self._rpath), "Can't seem to find {} anywhere".format(
                self._rpath)
            print('Finetuning model: load model from ...%s' %
                  self._rpath)
            self.action_model.load_state_dict(torch.load(
                self._rpath, map_location=torch.device('cpu')))
            self.optimizer_state_dict = None
            if resume:
                # create path for loading state dict
                optimizer_state_dict = join(
                    cfg.save_path, cfg.resume_path, f"model_save-optim.pt")
                self.optimizer_state_dict = torch.load(
                    optimizer_state_dict, map_location=torch.device('cpu'))
        else:
            self.optimizer_state_dict = None

        self.config = config
        self.train_cfg = config.train_cfg

        # move log path to here!
        print(
            f'{Fore.RED}\n----Done initializing Workspace, saving config.yaml to directory: {self.trainer.save_dir}----{Style.RESET_ALL}\n')

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
        loss_function = node_classification_loss

        self.trainer.train(model=self.action_model,
                           optimizer_state_dict=self.optimizer_state_dict,
                           loss_function=loss_function)

        print("Done training")
