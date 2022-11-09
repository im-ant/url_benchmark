import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from collections import deque
import datetime
import os
import random


os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import joblib  # for dumping and pickling sklearn models?
from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# from sklearn.cluster import MiniBatchKMeans

from dm_env import specs
import dmc
import utils

from dmc_benchmark import PRIMAL_TASKS


#from logger import Logger
# from replay_buffer import ReplayBufferStorage, make_replay_loader
#from video import TrainVideoRecorder, VideoRecorder

#torch.backends.cudnn.benchmark = True



class FilesDataset(Dataset):
    """
    Dataset object, index-based access of files to return as numpy objects
    """
    def __init__(self, storage_path, nstep, discount=0.99):
        self._storage_path = storage_path
        self._nstep = nstep
        self._discount = discount

        # self.exp_keys = ['observation', 'action', 'reward', 'discount']
        self._size = 0

        # ==
        # Get all files and  sort. Each file is experience from one episode
        self._episode_files = sorted(self._storage_path.glob('*.npz'),
                                     reverse=True)  # glob glob sort

    def __len__(self):
        return len(self._episode_files)

    def __getitem__(self, idx):
        eps_path = self._episode_files[idx]
        return np.load(eps_path)


class EpisodeFilesLoader:
    """
    Iterable Data Loader to sequentially loop through data in batches
    """
    def __init__(self, dataset, batch_size, torch=True):
        self._dataset = dataset
        self._file_idx = 0

        self._batch_size = batch_size
        self._torch = torch

        # Initialize data shape
        self._data_shape = np.shape(self._dataset[0]['observation'])[1:]

        # Iterate items
        self._cache_buf = np.empty((self._batch_size, *self._data_shape))
        self._cache_idx = 0

        self._b_queue = deque()  # TODO: customize maxlen

    @property
    def file_idx(self):
        return self._file_idx

    def _load_file(self, file_idx):
        """Loads a single file based on index of a Dataset object"""
        # Load file
        cur_f = self._dataset[file_idx]
        data_mat = cur_f['observation'][:-1]  # NOTE: ignore last entry in episode

        # Load the data in this file into cache
        d_idx_start = 0
        while d_idx_start < len(data_mat):
            # Figure out how much file data to load into cache
            cache_remain = self._batch_size - self._cache_idx
            d_idx = min(len(data_mat), d_idx_start+cache_remain)  # right index
            d_size = d_idx - d_idx_start  # how much data to load
            self._cache_buf[self._cache_idx : self._cache_idx+d_size] = \
                data_mat[d_idx_start : d_idx]  # Load data into cache buffer

            self._cache_idx = self._cache_idx + d_size
            d_idx_start = d_idx

            # If cache is full, put into queue and start new cache
            # TODO: bug check and edge case check
            if self._cache_idx == self._batch_size:
                self._b_queue.append(self._cache_buf)
                self._cache_buf = np.empty((self._batch_size,
                                            *self._data_shape))
                self._cache_idx = 0

    def _get_next_batch(self):
        # If queue is empty, load more files
        while len(self._b_queue) == 0:
            self._load_file(self._file_idx)
            self._file_idx += 1

            if self._file_idx == len(self._dataset):
                raise StopIteration

        batch = self._b_queue.popleft()
        if self._torch:
            batch = torch.from_numpy(batch).float()

        return batch


    def __iter__(self):
        while True:
            try:
                yield self._get_next_batch()
            except StopIteration:
                break

    def get_cache(self):
        """
        Get the current available batch in the cache. Not guaranteed to have
        size batch_size
        """
        cached_batch = self._cache_buf[:self._cache_idx]
        if self._torch:
            cached_batch = torch.from_numpy(cached_batch).float()
        return cached_batch


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # ===
        # Load dataset
        run_path = Path(self.cfg.base_dir) / self.cfg.run_dir
        dataset_path = run_path / self.cfg.data_path

        print('Dataset path:', dataset_path)
        self.file_dataset = FilesDataset(dataset_path, nstep=1, discount=0.99)
        print('Dataset:', self.file_dataset)

        # ==
        # Load encoder function
        self.agent = self.initialize_agent()
        # self.agent.optional_inits()  # optional additional init?
        # meta_specs = self.agent.get_meta_specs()  # get meta specs


    def initialize_agent(self):
        # ==
        # Initialize agent
        run_path = Path(self.cfg.base_dir) / self.cfg.run_dir

        if self.cfg.init_cfg_path is not None:
            cfg_path = run_path / self.cfg.init_cfg_path
            cfg = OmegaConf.load(cfg_path)
        else:
            cfg = self.cfg

        if 'task' in cfg:
            task = cfg.task
        else:
            task = PRIMAL_TASKS[cfg.domain]  # from reward free pretrain

        example_env = dmc.make(task, cfg.obs_type, cfg.frame_stack,
                               cfg.action_repeat, cfg.discretize_action,
                               cfg.seed)

        agent_cfg = cfg.agent
        agent_cfg.obs_type = cfg.obs_type
        agent_cfg.obs_shape = example_env.observation_spec().shape
        agent_cfg.action_shape = example_env.action_spec().shape
        agent_cfg.num_expl_steps = cfg.num_seed_frames // cfg.action_repeat
        agent_cfg.device = self.cfg.device

        agent = hydra.utils.instantiate(agent_cfg)

        # ==
        # Load pre-trained weights
        if self.cfg.model_path is not None:
            snapshot_path = run_path / self.cfg.model_path
            with snapshot_path.open('rb') as f:
                payload = torch.load(f, map_location=self.device)

            agent.init_from(payload['agent'])
            print('Initialized model from:', snapshot_path)

        return agent

    def encode_dataset(self):
        # Define the encoding function from the agent
        def _encode_fn(x):
            h = self.agent.encoder(x)
            return self.agent.actor.trunk(h)

        # Make the data loader
        loader = EpisodeFilesLoader(
            self.file_dataset, batch_size=self.cfg.loader_batch_size,
            torch=True,
        )
        print('Dataloader:', loader)

        # Iterate over files and save the encoded representations
        batch_queue = []
        for b_idx, batch_obs in enumerate(loader):
            print(f'Batch idx: {b_idx}. File idx: {loader.file_idx}/{len(self.file_dataset)}')

            batch_obs = batch_obs.to(self.device)
            batch_h = _encode_fn(batch_obs)  # encode
            batch_h = batch_h.detach().cpu().numpy()
            batch_queue.append(batch_h)

            if len(batch_queue) >= self.cfg.batch_in_out_file:
                ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
                self.save_np(batch_queue, f'{ts}_{b_idx}_{loader.file_idx}')
                batch_queue = []

        # Save the last batch from the cache
        last_batch_obs = loader.get_cache().to(self.device)
        last_batch_h = _encode_fn(last_batch_obs)  # encode
        last_batch_h = last_batch_h.detach().cpu().numpy()
        batch_queue.append(last_batch_h)

        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        self.save_np(batch_queue, f'{ts}_{b_idx}_{loader.file_idx}')

        print('Done encoding')

    def save_np(self, np_list, filename):
        mat = np.concatenate(np_list, axis=0)
        payload = {'x': mat}

        out_dir = Path(self.cfg.npz_out_dir)
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / filename

        np.savez_compressed(out_path, **payload)


@hydra.main(config_path='.', config_name='encode_dataset')
def main(cfg):
    # from modify_agent import Workspace as W  # TODO better import?
    root_dir = Path.cwd()
    workspace = Workspace(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()

    workspace.encode_dataset()
    #workspace.modify_agent()
    #workspace.save_snapshot()

if __name__ == '__main__':
    main()
