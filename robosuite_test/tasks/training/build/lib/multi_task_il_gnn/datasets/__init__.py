import random
import pickle as pkl
from multi_task_il.datasets.savers import Trajectory

SHUFFLE_RNG = 2843014334


def split_files(file_len, splits, mode='train'):
    assert sum(splits) == 1 and all(
        [0 <= s for s in splits]), "splits is not valid pdf!"

    order = [i for i in range(file_len)]
    random.Random(SHUFFLE_RNG).shuffle(order)
    pivot = int(len(order) * splits[0])
    if mode == 'train':
        order = order[:pivot]
    else:
        order = order[pivot:]
    return order


def load_traj(fname):
    if '.pkl' in fname:
        sample = pkl.load(open(fname, 'rb'))
        traj = sample['traj']
        if 'command' in sample.keys():
            command = sample['command']
        else:
            command = None
    else:
        raise NotImplementedError

    traj = traj
    return traj, command


def load_graph(fname):
    with open(fname, 'rb') as f:
        traj = pkl.load(f)
    return traj[0]
