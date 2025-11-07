from mmdet.models.backbones.swin import SwinTransformer
from mmengine.registry import init_default_scope
import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from multi_task_il.models import get_model
from multi_task_il.models.discrete_logistic import DiscreteMixLogistic
from multi_task_il.models.rep_modules import BYOLModule, ContrastiveModule
from multi_task_il.models.basic_embedding import TemporalPositionalEncoding
from einops import rearrange, repeat, parse_shape
from collections import OrderedDict
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from torchvision import models
import pickle
from multi_task_il.datasets.multi_task_datasets import MultiTaskPairedDataset
# mmdet moduls
from typing import (Any, Callable, Dict, Iterable, List, Optional, Sequence,
                    Tuple, Union)
from mmengine.registry import MODELS
from mmengine.config import Config, ConfigDict
from mmengine.runner.checkpoint import (_load_checkpoint,
                                        _load_checkpoint_to_model)
from mmengine.dataset import Compose
ConfigType = Union[Config, ConfigDict]


@MODELS.register_module()
class TargetObjDetector(nn.Module):

    def __init__(self,
                 demo_img_size: list = (200, 360),
                 agent_img_size: list = (100, 180),
                 backbone_chpt: str = None,
                 device: int = 0
                 ):
        super().__init__()

        # 1. Init Demo Backbone
        init_cfg = dict()
        init_cfg["type"] = 'Pretrained'
        init_cfg["checkpoint"] = backbone_chpt
        self._demo_backbone = SwinTransformer(pretrain_img_size=demo_img_size,
                                              in_channels=3,
                                              embed_dims=96,
                                              patch_size=4,
                                              window_size=7,
                                              mlp_ratio=4,
                                              depths=(2, 2, 6, 2),
                                              num_heads=(3, 6, 12, 24),
                                              strides=(4, 2, 2, 2),
                                              out_indices=(0, 1, 2, 3),
                                              qkv_bias=True,
                                              qk_scale=None,
                                              patch_norm=True,
                                              drop_rate=0.,
                                              attn_drop_rate=0.,
                                              drop_path_rate=0.1,
                                              use_abs_pos_embed=False,
                                              act_cfg=dict(type='GELU'),
                                              norm_cfg=dict(type='LN'),
                                              with_cp=False,
                                              pretrained=None,
                                              convert_weights=False,
                                              frozen_stages=-1,
                                              init_cfg=init_cfg)
        self._demo_backbone.to(f'cuda:{device}')

        # 2. Init Agent Backbone
        self._agent_backbone = SwinTransformer(pretrain_img_size=agent_img_size,
                                               in_channels=3,
                                               embed_dims=96,
                                               patch_size=4,
                                               window_size=7,
                                               mlp_ratio=4,
                                               depths=(2, 2, 6, 2),
                                               num_heads=(3, 6, 12, 24),
                                               strides=(4, 2, 2, 2),
                                               out_indices=(0, 1, 2, 3),
                                               qkv_bias=True,
                                               qk_scale=None,
                                               patch_norm=True,
                                               drop_rate=0.,
                                               attn_drop_rate=0.,
                                               drop_path_rate=0.1,
                                               use_abs_pos_embed=False,
                                               act_cfg=dict(type='GELU'),
                                               norm_cfg=dict(type='LN'),
                                               with_cp=False,
                                               pretrained=None,
                                               convert_weights=False,
                                               frozen_stages=-1,
                                               init_cfg=init_cfg)
        self._agent_backbone.to(f'cuda:{device}')

        #

    def forward(self, context, agent_obs):
        # 1. Compute demo and agent embedding
        demo_embedding = self._demo_backbone(context)[-1]
        agent_embedding = self._agent_backbone(agent_obs)[-1]

        # 2. Compute demo k,q,v and agent k,q,v


if __name__ == '__main__':
    import debugpy
    debugpy.listen(('0.0.0.0', 5678))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()

    checkpoint_path = "/home/frosa_loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training/multi_task_il/models/backbone_chpts/swin_small_patch4_window7_224.pth"
    print(f"Creating target object detector")
    target_obj_detector = TargetObjDetector(backbone_chpt=checkpoint_path)

    print(f"Fake execution")
    context = torch.from_numpy(np.random.rand(3, 200, 360))[
        None].float().to('cuda:0')
    agent_obs = torch.from_numpy(np.random.rand(3, 100, 180))[
        None].float().to('cuda:0')
    target_obj_detector(context, agent_obs)
