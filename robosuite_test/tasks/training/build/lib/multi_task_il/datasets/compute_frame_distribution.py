from multi_task_datasets import MultiTaskPairedDataset, DIYBatchSampler, collate_by_task
from omegaconf import DictConfig, OmegaConf
import hydra
from tqdm import tqdm
import numpy as np
from multiprocessing import cpu_count
from torch.utils.data import DataLoader

if __name__ == '__main__':

    # import debugpy
    # debugpy.listen(('0.0.0.0', 5678))
    # print("Waiting for debugger attach")
    # debugpy.wait_for_client()

    # 1. Load Training Dataset
    conf_file = OmegaConf.load(
        "/home/frosa_loc/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/baseline-1/Frame-Distribution-Batch128-1gpu-Attn2ly128-Act2ly256mix4-actCat-simclr128x512/config.yaml")
    # 2. Creatae dataset
    conf_file.dataset_cfg.mode = 'train'
    dataset = hydra.utils.instantiate(conf_file.dataset_cfg)
    # 3. Create traini sampler
    samplerClass = DIYBatchSampler
    train_sampler = samplerClass(
        task_to_idx=dataset.task_to_idx,
        subtask_to_idx=dataset.subtask_to_idx,
        tasks_spec=conf_file.dataset_cfg.tasks_spec,
        object_distribution_to_indx=dataset.object_distribution_to_indx,
        sampler_spec=conf_file.samplers,
        n_step=129536)
    train_loader = DataLoader(
        dataset,
        batch_sampler=train_sampler,
        num_workers=min(11, conf_file.get('loader_workers', cpu_count())),
        worker_init_fn=lambda w: np.random.seed(
            np.random.randint(2 ** 29) + w),
        collate_fn=collate_by_task
    )

    # 4. Iterate all over the dataset
    epochs = 3
    for e in range(epochs):
        frac = e / epochs
        print(e)
        for i, _ in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
            pass
    print(dataset._selected_target_frame_distribution_task_object_target_position.values())
