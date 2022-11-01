import warnings

import os

from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from dm_env import specs
import dmc
import utils

#from logger import Logger
#from replay_buffer import ReplayBufferStorage, make_replay_loader
#from video import TrainVideoRecorder, VideoRecorder

#torch.backends.cudnn.benchmark = True


def make_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape

    if type(action_spec) == specs.BoundedArray:
        cfg.action_shape = action_spec.shape
    elif type(action_spec) == specs.DiscreteArray:
        cfg.num_actions = action_spec.num_values
    else:
        raise NotImplementedError

    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # create environment
        self.train_env = dmc.make(cfg.task, cfg.obs_type, cfg.frame_stack,
                                  cfg.action_repeat, cfg.discretize_action,
                                  cfg.train_env_seed)


        # create agent
        print('Initializing agent...')
        self.agent = make_agent(cfg.obs_type,
                                self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)

        # initialize from pretrained
        pretrained_agent = self.load_snapshot()['agent']
        print(self.agent, pretrained_agent)
        self.agent.init_from(pretrained_agent)
        print('Loaded snapshop agent', pretrained_agent)

        # optional additional initialization for the agent
        self.agent.optional_inits()

        # get meta specs
        meta_specs = self.agent.get_meta_specs()

        self._global_step = None
        self._global_episode = None
        self.global_frame = self.cfg.snapshot_ts


    def modify_agent(self):
        """
        This is the customizable function to modify the agent and save it
        """

        # TODO: orthogonal init or other kinds of init

        print(self.agent.protos.weight.data.size())
        print(self.agent.protos.weight.data[0:3, 0:3])
        print(torch.norm(self.agent.protos.weight.data, p=2, dim=1)[:10])
        print()

        key_init = '2d_gridmesh'
        if key_init == 'orthonormal':
            # TODO: modify this for protos instead of policy head
            nn.init.orthogonal_(self.agent.protos.weight.data)
            C = self.agent.protos.weight.data.clone()
            C = F.normalize(C, dim=1, p=2)
            self.agent.protos.weight.data.copy_(C)
        elif key_init == '2d_gridmesh':
            x_pts = torch.linspace(-0.3, 0.3, 23)  # mesh grid even covering
            xy_grid = torch.cartesian_prod(x_pts, x_pts).to(self.device)
            print('Grid size', xy_grid.size())

            s_grid = self.agent.encoder(xy_grid)
            s_grid = self.agent.predictor(s_grid)
            # s_grid = F.normalize(s, dim=1, p=2)
            print('projected size',s_grid.size())

            C = s_grid[:self.agent.protos.weight.data.size(0), :]
            self.agent.protos.weight.data.copy_(C)

        else:
            raise NotImplementedError

        print(self.agent.protos.weight.data.size())
        print(self.agent.protos.weight.data[0:3, 0:3])
        print(torch.norm(self.agent.protos.weight.data, p=2, dim=1)[:10])
        print()


    def load_snapshot(self):
        snapshot_base_dir = Path(self.cfg.snapshot_base_dir)
        if self.cfg.snapshot_child_dir is None:
            domain, _ = self.cfg.task.split('_', 1)
            snapshot_dir = snapshot_base_dir / self.cfg.obs_type / domain / self.cfg.agent.name
        else:
            snapshot_dir = snapshot_base_dir / Path(self.cfg.snapshot_child_dir)

        def try_load(seed):
            if self.cfg.snapshot_child_dir is None:
                snapshot = snapshot_dir / str(
                    seed) / f'snapshot_{self.cfg.snapshot_ts}.pt'
            else:
                snapshot = snapshot_dir / f'snapshot_{self.cfg.snapshot_ts}.pt'
            if not snapshot.exists():
                return None
            with snapshot.open('rb') as f:
                payload = torch.load(f, map_location=self.device)
            return payload

        # TODO: should have config for which pretraining seed to use, rather
        #       than use the same seed as fine-tuning

        # try to load current seed
        payload = try_load(self.cfg.seed)
        if payload is not None:
            return payload
        # otherwise try all the seed between 1-10
        for seed in range(1, 11):
            payload = try_load(seed)
            if payload is not None:
                return payload
        # otherwise throw error
        ssdp = snapshot_dir / '[1-11]' / f'snapshot_{self.cfg.snapshot_ts}.pt'
        raise RuntimeError(f'Did not find snapshot at: {ssdp}')

    def save_snapshot(self):
        # TODO: see this works still
        snapshot_dir = self.work_dir / Path(self.cfg.snapshot_out_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        snapshot = snapshot_dir / f'snapshot_{self.global_frame}.pt'
        keys_to_save = ['agent', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)
            print('Saved to:', snapshot)


@hydra.main(config_path='.', config_name='finetune')
def main(cfg):
    from modify_agent import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()

    workspace.modify_agent()
    workspace.save_snapshot()

if __name__ == '__main__':
    main()
