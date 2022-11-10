import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

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
from torch.utils.data import Dataset

# from sklearn.cluster import KMeans, SpectralClustering

import utils
from logger import Logger




#from logger import Logger
# from replay_buffer import ReplayBufferStorage, make_replay_loader
#from video import TrainVideoRecorder, VideoRecorder

#torch.backends.cudnn.benchmark = True



class ExperienceDataset(Dataset):
    # TODO: delete this whole class?
    def __init__(self, storage_path, nstep, discount=0.99):
        self._storage_path = storage_path
        self._nstep = nstep
        self._discount = discount

        self.exp_keys = ['observation', 'action', 'reward', 'discount']
        self._size = 0

        # ==
        # Initialization

        # Get all files and  sort. Each file is experience from one episode
        self._episode_fns = sorted(self._storage_path.glob('*.npz'),
                                   reverse=True)  # glob glob sort

        # Construct index book for later sampling
        self._idx_paths = []  # list containing the file path for each data idx
        self._n_exps_before = {}
        for eps_path in self._episode_fns:
            cur_eps = np.load(eps_path)

            self._n_exps_before[eps_path] = self._size

            # Subtract 1: first index is a dummy (?) for transition
            cur_eps_len = len(cur_eps['reward']) - 1  # loading reward is fast

            self._idx_paths.extend([eps_path] * cur_eps_len)
            self._size += cur_eps_len

    def __len__(self):
        return self._size  # TODO: check this is kosher

    def __getitem__(self, idx):
        """
        Get individual items based on their index
        """
        # Load episode
        eps_path = self._idx_paths[idx]
        episode = np.load(eps_path, mmap_mode='r')

        # Index in episode to sample (plus 1 to account for transition)
        in_eps_idx = idx - self._n_exps_before[eps_path] + 1

        # Get information
        obs = episode['observation'][in_eps_idx - 1]
        action = episode['action'][in_eps_idx]
        next_obs = episode['observation'][in_eps_idx + self._nstep - 1]
        reward = np.zeros_like(episode['reward'][in_eps_idx])
        discount = np.ones_like(episode['discount'][in_eps_idx])
        for i in range(self._nstep):
            step_reward = episode['reward'][in_eps_idx + i]
            reward += discount * step_reward
            discount *= episode['discount'][in_eps_idx + i] * self._discount

        return (obs, action, reward, discount, next_obs)




def _worker_init_fn(worker_id):
    # TODO: delete this whole thing?
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)

def _loggable_attr(obj, attr):
    if attr.startswith('_'):
        return False
    attr_val = getattr(obj, attr)
    if isinstance(attr_val, float):
        return True
    if isinstance(attr_val, int):
        return True
    return False


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # ===
        # Load dataset
        base_path = Path(self.cfg.base_dir)
        self.data_dir_path = base_path / self.cfg.data_path
        print('Data dir path:', self.data_dir_path)

        self.data_X = self.load_dataset()
        print('Data shape:', np.shape(self.data_X))

        # Load algorithm
        self.clustering = hydra.utils.instantiate(cfg.clustering)
        print(self.clustering)

        #
        self.logger = Logger(self.work_dir, use_tb=False, use_wandb=False)
        self.timer = utils.Timer()

        # ??
        self.global_step = 1
        #self._global_episode = None
        #self.global_frame = self.cfg.snapshot_ts

    def load_dataset(self):
        data_files = sorted(self.data_dir_path.glob('*.npz'), reverse=True)

        data_list = []
        for f_path in data_files:
            cur_f = np.load(f_path)
            data_list.append(cur_f['x'])

        data_mat = np.concatenate(data_list, axis=0)

        # if subsample option
        if self.cfg.subsample_data:
            subsampl_idxs = np.random.choice(len(data_mat),
                                             size=self.cfg.subsample_size,
                                             replace=False)
            data_mat = data_mat[subsampl_idxs]

        return data_mat

    def train(self):
        # Fit to data
        self.clustering.fit(self.data_X)

        # Log
        elapsed_time, total_time = self.timer.reset()
        with self.logger.log_and_dump_ctx(step=self.global_step,
                                          ty='train') as log:
            log('step', self.global_step)
            log('total_time', total_time)
            log('episode_length', len(self.data_X))

            #log('fps', 1.)
            #log('episode_reward', 2.)
            #log('episode', None)
            #log('buffer_size', 100.)

            for attr in dir(self.clustering):
                if _loggable_attr(self.clustering, attr):
                    log(attr, getattr(self.clustering, attr))

        self.save_snapshot()

    def save_snapshot(self):
        snapshot_dir = self.work_dir / Path(self.cfg.snapshot_out_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)

        out_filename = f'clus_{self.global_step}_{len(self.data_X)}.pkl'
        out_path = snapshot_dir / out_filename

        joblib.dump(self.clustering, out_path)
        

    def train_old(self):

        # TODO: testing below
        kmeans = MiniBatchKMeans(n_clusters=128, random_state=0, batch_size=512)

        for b_idx, batch in enumerate(iter(self.loader)):
            obs, action,reward, discount, next_obs = utils.to_torch(
                batch, self.cfg.device)
            states = self.agent.encoder(obs)
            actor_phis = self.agent.actor.trunk(states)

            kmeans.partial_fit(actor_phis.detach().cpu().numpy())
            print(b_idx, actor_phis.size(), kmeans.inertia_)

            if b_idx % 10 == 0:
                joblib.dump(kmeans, "kmeans_model.pkl")




@hydra.main(config_path='.', config_name='cluster_dataset')
def main(cfg):
    root_dir = Path.cwd()
    workspace = Workspace(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()

    workspace.train()
    #workspace.modify_agent()
    #workspace.save_snapshot()

if __name__ == '__main__':
    main()
