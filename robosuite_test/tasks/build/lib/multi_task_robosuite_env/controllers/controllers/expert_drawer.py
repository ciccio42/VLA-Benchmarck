import sys
from pathlib import Path

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))
import numpy as np
from multi_task_il.datasets import Trajectory
import pybullet as p
from pyquaternion import Quaternion
import random
from multi_task_robosuite_env.custom_ik_wrapper import normalize_action
from robosuite import load_controller_config
from robosuite.utils.transform_utils import quat2axisangle
from robosuite.utils import RandomizationError
import torch
import os
# import mujoco_py
import robosuite.utils.transform_utils as T
import multi_task_robosuite_env.utils as utils
from multi_task_robosuite_env import get_env
# in case rebuild is needed to use GPU render: sudo mkdir -p /usr/lib/nvidia-000
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
# pip uninstall mujoco_py; pip install mujoco_py

import copy
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
nut_assembly_logger = logging.getLogger(name="NutAssemblyLogger")


def _clip_delta(delta, max_step=0.015):
    norm_delta = np.linalg.norm(delta)
    if norm_delta < max_step:
        return delta
    return delta / norm_delta * max_step


class DrawerController:
    pass
