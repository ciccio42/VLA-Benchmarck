from torch.utils.data import Dataset, Sampler, SubsetRandomSampler, RandomSampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from typing import Optional
from collections import OrderedDict
import math
import random

class DIYBatchSampler(DistributedSampler):
    """
    Customize any possible combination of both task families and sub-tasks in a batch of data.
    """

    def __init__(
        self,
        task_to_idx,
        subtask_to_idx,
        object_distribution_to_indx,
        sampler_spec=dict(),
        tasks_spec=dict(),
        n_step=0,
        dataset: Dataset = None,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 42, 
    ):
        
        self.drop_last = sampler_spec.get('drop_last', False)
        self.shuffle = sampler_spec.get('shuffle', False)
        self.num_samples_per_task = OrderedDict()
        self.rank = rank
        
        # key: process index
        self.task_samplers_per_process = OrderedDict()
        self.task_iterators_per_process = OrderedDict()
        self.task_info_per_process = OrderedDict()
        self.idx_map = OrderedDict()
        for i in range(num_replicas):
            # key: task name
            self.task_samplers_per_process[i] = OrderedDict()
            self.task_iterators_per_process[i] = OrderedDict()
            self.task_info_per_process[i] = OrderedDict()
        
        
        self.balancing_policy = sampler_spec.get('balancing_policy', 0)
        self.object_distribution_to_indx = object_distribution_to_indx
        self.num_step = n_step
        self.num_replicas = num_replicas
        
        super(DIYBatchSampler, self).__init__(dataset, 
                                              num_replicas, 
                                              rank, 
                                              self.shuffle,
                                              seed, 
                                              self.drop_last)
        
        batch_size = sampler_spec.get('batch_size', 30)

        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(self.drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(self.drop_last))

        # for each task compute the number of samples distributed for each process
        for spec in tasks_spec:
            task_name = spec.name
            # global number of indices for task_name
            idxs = task_to_idx.get(task_name)
            
            if not self.drop_last:
                self.num_samples_per_task[task_name] = math.ceil(len(task_to_idx.get(task_name)) / num_replicas)
            else:
                raise Exception("Not implemented yet")
               
           
            # compute for each process the indices that are assigned to it
            for rank_indx in range(num_replicas):
                # uniformly draw from union of all sub-tasks
                self.task_samplers_per_process[rank_indx][task_name] = OrderedDict(
                {'all_sub_tasks': SubsetRandomSampler(idxs[rank_indx : len(idxs) : self.num_replicas])})  
                self.task_iterators_per_process[rank_indx][task_name] =  OrderedDict(
                {'all_sub_tasks': iter(SubsetRandomSampler(idxs[rank_indx : len(idxs) : self.num_replicas]))})

            num_loaded_sub_tasks = len(subtask_to_idx[task_name].keys())
            first_id = list(subtask_to_idx[task_name].keys())[0]    
            if not self.drop_last:
                sub_task_size = math.ceil(len(subtask_to_idx[task_name].get(first_id))/ num_replicas)
            else:
                raise Exception("Not implemented yet")
            
            print("Task {} loaded {} subtasks, starting from {}, should all have sizes {}".format(
                    task_name, num_loaded_sub_tasks, first_id, sub_task_size))
        
            # for each sub-task compute the number of samples distributed for each process
            for sub_task, sub_idxs in subtask_to_idx[task_name].items():
                if self.balancing_policy == 1 and self.object_distribution_to_indx != None:
                    raise Exception("Balancing policy not implemented yet") 
                else:
                    for rank_indx in range(num_replicas):
                        self.task_samplers_per_process[rank_indx][task_name][sub_task] = SubsetRandomSampler(
                            sub_idxs[rank_indx : len(sub_idxs) : num_replicas])
                        self.task_iterators_per_process[rank_indx][task_name][sub_task] = iter(
                            SubsetRandomSampler(sub_idxs[rank_indx : len(sub_idxs) : num_replicas]))
                    
            for rank_id in range(self.num_replicas):
                curr_task_info = {
                'size':         self.num_samples_per_task[task_name],
                'n_tasks':      len(subtask_to_idx[task_name].keys()),
                'sub_id_to_name': {i: name for i, name in enumerate(subtask_to_idx[task_name].keys())},
                'traj_per_subtask': sub_task_size,
                'sampler_len': -1  # to be decided below
                }            
                self.task_info_per_process[rank_id][task_name] = curr_task_info


        self.n_tasks = len(self.task_samplers_per_process[rank].keys())
        n_total = sum([info['size'] for info in self.task_info_per_process[rank].values()])

        for rank_id in range(num_replicas):
            idx = 0
            # assign the batch indices to the sub-tasks
            for spec in tasks_spec:
                name = spec.name
                _ids = spec.get('task_ids', None)
                n = spec.get('n_per_task', None)
                assert (
                    _ids and n), 'Must specify which subtask ids to use and how many is contained in each batch'
                info = self.task_info_per_process[rank_id][name]
                subtask_names = info.get('sub_id_to_name')
                for subtask in subtask_names.values():
                    for _ in range(n):
                        # position idx of batch is a sample of task [name] subtask [subtask]
                        self.idx_map[idx] = (name, subtask)
                        idx += 1
                    sub_length = int(info['traj_per_subtask'] / n)
                    self.task_info_per_process[rank_id][name]['sampler_len'] = max(
                        sub_length, self.task_info_per_process[rank_id][name]['sampler_len'])
        
        # print("Index map:", self.idx_map)
        # number of steps that I need for covering all the couple (demo, agent)
        self.max_len = max([info['sampler_len']
                            for info in self.task_info_per_process[rank].values()])
        print('Max length for sampler iterator:', self.max_len)
        

        assert idx == batch_size, "The constructed batch size {} doesn't match desired {}".format(
            idx, batch_size)
        self.batch_size = idx
        
        print("Shuffling to break the task ordering in each batch? ", self.shuffle)

    def __iter__(self):
        batch = []
        for i in range(self.max_len):
            # for each sample in the batch
            for idx in range(self.batch_size):
                (name, sub_task) = self.idx_map[idx]

                if self.balancing_policy == 1 and self.object_distribution_to_indx != None:
                    slot_indx = idx % len(self.task_samplers_per_process[self.rank][name][sub_task])
                    # take one sample for the current task, sub_task, and slot
                    sampler = self.task_samplers_per_process[self.rank][name][sub_task][slot_indx]
                    iterator = self.task_iterators_per_process[self.rank][name][sub_task][slot_indx]
                    try:
                        batch.append(next(iterator))
                    except StopIteration:  # print('early sstop:', i, name)
                        # re-start the smaller-sized tasks
                        print("Stop iteration")
                        iterator = iter(sampler)
                        batch.append(next(iterator))
                        self.task_iterators_per_process[self.rank][name][sub_task][slot_indx] = iterator
                else:
                    # print(name, sub_task)
                    sampler = self.task_samplers_per_process[self.rank][name][sub_task]
                    iterator = self.task_iterators_per_process[self.rank][name][sub_task]
                    try:
                        batch.append(next(iterator))
                    except StopIteration:  # print('early sstop:', i, name)
                        # re-start the smaller-sized tasks
                        print("Stop Iteration")
                        iterator = iter(sampler)
                        batch.append(next(iterator))
                        self.task_iterators_per_process[self.rank][name][sub_task] = iterator

            if len(batch) == self.batch_size:
                if self.shuffle:
                    random.shuffle(batch)
                yield batch
                batch = []
            if len(batch) > 0 and not self.drop_last:
                if self.shuffle:
                    random.shuffle(batch)
                yield batch

    def __len__(self):
        # Since different task may have different data sizes,
        # define total length of sampler as number of iterations to
        # exhaust the last task
        return self.max_len


class TrajectoryBatchSampler(DistributedSampler):

    def __init__(
        self,
        dataset,
        agent_files,
        task_to_idx,
        subtask_to_idx,
        demo_task_to_idx,
        demo_subtask_to_idx,
        object_distribution_to_indx,
        sampler_spec=dict(),
        tasks_spec=dict(),
        n_step=0,
        epoch_steps=0,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 42, 
    ):

        batch_size = sampler_spec.get('batch_size', 30)
        drop_last = sampler_spec.get('drop_last', False)
        self.rank = rank

        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        self.shuffle = sampler_spec.get('shuffle', False)

        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        self.balancing_policy = sampler_spec.get('balancing_policy', 0)
        self.num_step = n_step

        # key: process index
        self.agent_sampler_per_process= OrderedDict()
        self.demo_sampler_per_process = OrderedDict()
        self.agent_iterator_per_process = OrderedDict()
        self.demo_iterator_per_process = OrderedDict()
        
        self.task_info_per_process = OrderedDict()
        self.frames_samplers_per_process = OrderedDict()
        self.frames_iterators_per_process = OrderedDict()
        self.idx_map = OrderedDict()
        for i in range(num_replicas):
            # key: task name
            self.agent_sampler_per_process[i] = OrderedDict()
            self.demo_sampler_per_process[i] = OrderedDict()
            self.agent_iterator_per_process[i] = OrderedDict()
            self.demo_iterator_per_process[i] = OrderedDict()
            
            self.task_info_per_process[i] = OrderedDict()
            self.frames_samplers_per_process[i] = OrderedDict()
            self.frames_iterators_per_process[i] = OrderedDict()
        
        self.num_step = n_step
        self.num_replicas = num_replicas
        print(f"Num replicas: {num_replicas}")
        super(TrajectoryBatchSampler, self).__init__(dataset, 
                                              num_replicas, 
                                              rank, 
                                              self.shuffle,
                                              seed, 
                                              drop_last)
        
        
        # Create sampler for agent trajectories
        for spec in tasks_spec:
            
            task_name = spec.name
            
            # global number of indices for task_name
            idxs = task_to_idx.get(task_name)
            demo_idxs = demo_task_to_idx.get(task_name)
            
            self.agent_sampler_per_process[rank][task_name] = OrderedDict(
                {'all_sub_tasks': SubsetRandomSampler(idxs[rank : len(idxs) : num_replicas])})
            self.demo_sampler_per_process[rank][task_name] = OrderedDict(
                {'all_sub_tasks': iter(SubsetRandomSampler(demo_idxs[rank : len(idxs) : num_replicas]))})    
            
            self.agent_iterator_per_process[rank][task_name] = OrderedDict(
                {'all_sub_tasks': iter(SubsetRandomSampler(idxs[rank : len(idxs) : num_replicas]) )})
            self.demo_iterator_per_process[rank][task_name] = OrderedDict(
                {'all_sub_tasks': iter(SubsetRandomSampler(demo_idxs[rank : len(idxs) : num_replicas]))})
            
            
            self.frames_samplers_per_process[rank][task_name] = OrderedDict()
            self.frames_iterators_per_process[rank][task_name] = OrderedDict()
                        
            # for each sub-task compute the number of samples 
            for sub_task, sub_idxs in subtask_to_idx[task_name].items():
                demo_sub_idxs = demo_subtask_to_idx[task_name].get(sub_task)
                
                self.frames_samplers_per_process[rank][task_name][sub_task] = OrderedDict()
                self.frames_iterators_per_process[rank][task_name][sub_task] = OrderedDict()
                
                if self.balancing_policy == 1 and self.object_distribution_to_indx != None:
                    raise Exception("Balancing policy not implemented yet") 
                else:
                    self.agent_sampler_per_process[rank][task_name][sub_task] = SubsetRandomSampler(sub_idxs[rank : len(sub_idxs) : num_replicas]) # indx of sub-task trjs
                    self.agent_iterator_per_process[rank][task_name][sub_task] = iter(SubsetRandomSampler(sub_idxs[rank : len(sub_idxs) : num_replicas]))
                    
                    self.demo_sampler_per_process[rank][task_name][sub_task] = SubsetRandomSampler(demo_sub_idxs[rank:len(demo_sub_idxs):num_replicas]) # indx of sub-task trjs
                    self.demo_iterator_per_process[rank][task_name][sub_task] = iter(SubsetRandomSampler(demo_sub_idxs[rank:len(demo_sub_idxs):num_replicas]))
                    
                    
                    for sub_indx in sub_idxs:
                        self.frames_samplers_per_process[rank][task_name][sub_task][sub_indx] = RandomSampler(range(agent_files[sub_indx][-1])) # indx of frames
                        self.frames_iterators_per_process[rank][task_name][sub_task][sub_indx] = iter(RandomSampler(range(agent_files[sub_indx][-1])))
                    
                    
            curr_task_info = {
                'size':         len(idxs), # size of the task
                'n_tasks':      len(subtask_to_idx[task_name].keys()),
                'sub_id_to_name': {i: name for i, name in enumerate(subtask_to_idx[task_name].keys())},
                'traj_per_subtask': len(subtask_to_idx[task_name].get(list(subtask_to_idx[task_name].keys())[0]))/num_replicas,
                'sampler_len': -1  # to be decided below
                }            

            self.task_info_per_process[rank][task_name] = curr_task_info
            
        self.n_tasks = len(self.agent_sampler_per_process[rank].keys())
        n_total = sum([info['size'] for info in self.task_info_per_process[rank].values()])
        idx = 0
        # assign the batch indices to the sub-tasks
        for spec in tasks_spec:
            name = spec.name
            _ids = spec.get('task_ids', None)
            n = spec.get('n_per_task', None)
            assert (
                _ids and n), 'Must specify which subtask ids to use and how many is contained in each batch'
            info = self.task_info_per_process[rank][name]
            subtask_names = info.get('sub_id_to_name')
            for subtask in subtask_names.values():
                for _ in range(n):
                    # position idx of batch is a sample of task [name] subtask [subtask]
                    self.idx_map[idx] = (name, subtask)
                    idx += 1
                sub_length = int(info['traj_per_subtask'] / n)
                self.task_info_per_process[rank][name]['sampler_len'] = max(
                    sub_length, self.task_info_per_process[rank][name]['sampler_len'])
        
        # # print("Index map:", self.idx_map)
        # # number of steps that I need for covering all the couple (demo, agent)
        self.max_len = epoch_steps
        #max([info['sampler_len']
                        #     for info in self.task_info.values()])
        print('Max length for sampler iterator:', self.max_len)
        

        assert idx == batch_size, "The constructed batch size {} doesn't match desired {}".format(
            idx, batch_size)
        self.batch_size = idx
        
        print("Shuffling to break the task ordering in each batch? ", self.shuffle)

    def __iter__(self):
        """Given task families A,B,C, each has sub-tasks A00, A01,...
        Fix a total self.batch_size, sample different numbers of datapoints from
        each task"""
        batch = []
        
        # In one epch I must to see all the frames of each trajectory
        
        
        for i in range(self.max_len):
            batch = []
            
            # for each sample in the batch
            for idx in range(self.batch_size):
                    
                (name, sub_task) = self.idx_map[idx]
                # sample indx of sub-task 
                sampler = self.agent_sampler_per_process[self.rank][name][sub_task]
                iterator = self.agent_iterator_per_process[self.rank][name][sub_task]
                try:
                    sample_idx = next(iterator)
                except StopIteration:  # print('early sstop:', i, name)
                    # re-start the smaller-sized tasks
                    # print("Stop Iteration")
                    iterator = iter(sampler)
                    sample_idx = next(iterator)
                    self.agent_iterator_per_process[self.rank][name][sub_task] = iterator
                
                # sample indx of demo
                demo_sampler = self.demo_sampler_per_process[self.rank][name][sub_task]
                demo_iterator = self.demo_iterator_per_process[self.rank][name][sub_task]
                try:
                    demo_idx = next(demo_iterator)
                except StopIteration:
                    # print("Stop Iteration")
                    demo_iterator = iter(demo_sampler)
                    demo_idx = next(demo_iterator)
                    self.demo_iterator_per_process[self.rank][name][sub_task] = demo_iterator
                       
                # sample indx of frames
                frames_sampler = self.frames_samplers_per_process[self.rank][name][sub_task][sample_idx]
                frames_iterator = self.frames_iterators_per_process[self.rank][name][sub_task][sample_idx]
                try:
                    frame_idx = next(frames_iterator)
                except StopIteration:
                    print("Reset iterator for frames")
                    frames_iterator = iter(frames_sampler)
                    frame_idx = next(frames_iterator)
                    self.frames_iterators_per_process[self.rank][name][sub_task][sample_idx] = frames_iterator
                                
                batch.append([demo_idx, sample_idx, frame_idx])

            if len(batch) == self.batch_size:
                if self.shuffle:
                    random.shuffle(batch)
                yield batch
                batch = []
            if len(batch) > 0 and not self.drop_last:
                if self.shuffle:
                    random.shuffle(batch)
                yield batch

    def __len__(self):
        # Since different task may have different data sizes,
        # define total length of sampler as number of iterations to
        # exhaust the last task
        print(f"Sampler max-len {self.max_len}")
        return self.max_len


class AgentBatchSampler(Sampler):

    def __init__(
        self,
        agent_task_to_idx,
        agent_subtask_to_idx,
        sampler_spec=dict(),
        tasks_spec=dict(),
        n_step=0,
        epoch_steps=0
    ):

        batch_size = sampler_spec.get('batch_size', 30)
        drop_last = sampler_spec.get('drop_last', False)

        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        self.shuffle = sampler_spec.get('shuffle', False)

        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        self.task_info = OrderedDict()
        self.balancing_policy = sampler_spec.get('balancing_policy', 0)
        self.num_step = n_step

        # Create sampler for agent trajectories
        self.agent_task_samplers = OrderedDict()
        self.agent_task_iterators = OrderedDict()
        self.agent_task_to_idx = agent_task_to_idx
        self.agent_subtask_to_idx = agent_subtask_to_idx
        for spec in tasks_spec:
            task_name = spec.name
            idxs = agent_task_to_idx.get(task_name)
            self.agent_task_samplers[task_name] = OrderedDict(
                {'all_sub_tasks': RandomSampler(data_source=idxs,
                                                replacement=True,
                                                num_samples=1)})  # uniformly draw from union of all sub-tasks
            self.agent_task_iterators[task_name] = OrderedDict(
                {'all_sub_tasks': iter(RandomSampler(data_source=idxs,
                                                     replacement=True,
                                                     num_samples=1))})
            assert task_name in agent_subtask_to_idx.keys(), \
                'Mismatch between {} task idxs and subtasks!'.format(
                    task_name)
            num_loaded_sub_tasks = len(agent_subtask_to_idx[task_name].keys())
            first_id = list(agent_subtask_to_idx[task_name].keys())[0]

            sub_task_size = len(agent_subtask_to_idx[task_name].get(first_id))
            print("Task {} loaded {} subtasks, starting from {}, should all have sizes {}".format(
                task_name, num_loaded_sub_tasks, first_id, sub_task_size))

            for sub_task, sub_idxs in agent_subtask_to_idx[task_name].items():

                self.agent_task_samplers[task_name][sub_task] = RandomSampler(
                    data_source=sub_idxs,
                    replacement=True,
                    num_samples=1)
                assert len(sub_idxs) == sub_task_size, \
                    'Got uneven data sizes for sub-{} under the task {}!'.format(
                        sub_task, task_name)
                self.agent_task_iterators[task_name][sub_task] = iter(
                    RandomSampler(data_source=sub_idxs,
                                  replacement=True,
                                  num_samples=1))
                # print('subtask indexs:', sub_task, max(sub_idxs))
            curr_task_info = {
                'size':         len(idxs),
                'n_tasks':      len(agent_subtask_to_idx[task_name].keys()),
                'sub_id_to_name': {i: name for i, name in enumerate(agent_subtask_to_idx[task_name].keys())},
                'traj_per_subtask': sub_task_size,
                'sampler_len': -1  # to be decided below
            }
            self.task_info[task_name] = curr_task_info

        # Create sampler for demo trajectories
        self.demo_task_samplers = OrderedDict()
        self.demo_task_iterators = OrderedDict()
        for spec in tasks_spec:
            task_name = spec.name
            self.demo_task_samplers[task_name] = OrderedDict(
                {'all_sub_tasks': RandomSampler(data_source=idxs,
                                                replacement=True,
                                                num_samples=1)})  # uniformly draw from union of all sub-tasks
            self.demo_task_iterators[task_name] = OrderedDict(
                {'all_sub_tasks': iter(RandomSampler(data_source=idxs,
                                                     replacement=True,
                                                     num_samples=1))})

            print("Task {} loaded {} subtasks, starting from {}, should all have sizes {}".format(
                task_name, num_loaded_sub_tasks, first_id, sub_task_size))

        n_tasks = len(self.agent_task_samplers.keys())
        n_total = sum([info['size'] for info in self.task_info.values()])

        self.idx_map = OrderedDict()
        idx = 0
        for spec in tasks_spec:
            name = spec.name
            _ids = spec.get('task_ids', None)
            _skip_ids = spec.get('skip_ids', [])
            n = spec.get('n_per_task', None)
            assert (
                _ids and n), 'Must specify which subtask ids to use and how many is contained in each batch'
            info = self.task_info[name]
            subtask_names = info.get('sub_id_to_name')
            for subtask in subtask_names.values():
                for _ in range(n):
                    # position idx of batch is a sample of task [name] subtask [subtask]
                    self.idx_map[idx] = (name, subtask)
                    idx += 1
                sub_length = int(info['traj_per_subtask'] / n)
                self.task_info[name]['sampler_len'] = max(
                    sub_length, self.task_info[name]['sampler_len'])
        # print("Index map:", self.idx_map)

        self.max_len = epoch_steps
        print('Max length for sampler iterator:', self.max_len)
        self.n_tasks = n_tasks
        self.epoch_steps = epoch_steps

        assert idx == batch_size, "The constructed batch size {} doesn't match desired {}".format(
            idx, batch_size)
        self.batch_size = idx
        self.drop_last = drop_last
        print("Shuffling to break the task ordering in each batch? ", self.shuffle)

    def __iter__(self):
        """Given task families A,B,C, each has sub-tasks A00, A01,...
        Fix a total self.batch_size, sample different numbers of datapoints from
        each task"""
        batch = []
        print("Reset agent_demo_pair")
        agent_demo_pair = dict()
        for i in range(self.max_len):
            batch = []
            if i % self.epoch_steps == 0:
                print("Reset agent_demo_pair")
                agent_demo_pair = dict()

            # for each sample in the batch
            for idx in range(self.batch_size):
                (name, sub_task) = self.idx_map[idx]

                agent_sampler = self.agent_task_samplers[name][sub_task]
                agent_iterator = self.agent_task_iterators[name][sub_task]

                try:
                    agent_indx = self.agent_subtask_to_idx[name][sub_task][next(
                        agent_iterator)]
                except StopIteration:  # print('early sstop:', i, name)
                    # re-start the smaller-sized tasks
                    agent_iterator = iter(agent_sampler)
                    agent_indx = self.agent_subtask_to_idx[name][sub_task][next(
                        agent_iterator)]
                    self.agent_task_iterators[name][sub_task] = agent_iterator

                # check if the agent_indx has already sampled
                # if agent_demo_pair.get(agent_indx, None) is None:
                    demo_sampler = self.demo_task_samplers[name][sub_task]
                    demo_iterator = self.demo_task_iterators[name][sub_task]
                    # new agent_indx in epoch
                    # sample demo for current
                    try:
                        demo_indx = self.demo_subtask_to_idx[name][sub_task][next(
                            demo_iterator)]
                    except StopIteration:  # print('early sstop:', i, name)
                        # re-start the smaller-sized tasks
                        demo_iterator = iter(demo_sampler)
                        demo_indx = self.demo_subtask_to_idx[name][sub_task][next(
                            demo_iterator)]
                        self.demo_task_iterators[name][sub_task] = demo_iterator
                    agent_demo_pair[agent_indx] = demo_indx

                batch.append([agent_indx, agent_demo_pair[agent_indx]])

            if len(batch) == self.batch_size:
                if self.shuffle:
                    random.shuffle(batch)
                yield batch
                batch = []
            if len(batch) > 0 and not self.drop_last:
                if self.shuffle:
                    random.shuffle(batch)
                yield batch

    def __len__(self):
        # Since different task may have different data sizes,
        # define total length of sampler as number of iterations to
        # exhaust the last task
        print(f"Sampler max-len {self.max_len}")
        return self.max_len
