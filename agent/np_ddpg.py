from collections import OrderedDict

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from agent.ddpg import Encoder, Actor, DDPGAgent
from modules.neural_dictionary import NeuralKNN


class NonParametricCritic(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim,
                 feature_dim, hidden_dim, memory_overwrite,
                 value_head_cfg, device):
        super().__init__()

        self.obs_type = obs_type
        self.memory_overwrite = memory_overwrite

        if obs_type == 'pixels':
            # for pixels actions will be added after trunk
            self.trunk = nn.Sequential(nn.Linear(obs_dim, feature_dim),
                                       nn.LayerNorm(feature_dim), nn.Tanh())
            trunk_dim = feature_dim + action_dim
        else:
            # for states actions come in the beginning
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim), nn.Tanh())
            trunk_dim = hidden_dim

        def make_q_neck():
            q_layers = []
            if obs_type == 'pixels':
                q_layers += [
                    nn.Linear(trunk_dim, hidden_dim),
                    nn.ReLU(inplace=True)
                ]
            q_layers += [nn.Identity()]
            return nn.Sequential(*q_layers)

        self.Q1_neck = make_q_neck()
        self.Q2_neck = make_q_neck()

        self.Q1_head = hydra.utils.instantiate(value_head_cfg).to(device)
        self.Q2_head = hydra.utils.instantiate(value_head_cfg).to(device)

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        q1, q2, info1, info2 = self.detailed_forward(obs, action)
        return q1, q2

    def detailed_forward(self, obs, action):
        inpt = obs if self.obs_type == 'pixels' else torch.cat([obs, action],
                                                               dim=-1)
        h = self.trunk(inpt)
        h = torch.cat([h, action], dim=-1) if self.obs_type == 'pixels' else h

        phis_1 = self.Q1_neck(h)
        phis_2 = self.Q2_neck(h)

        if hasattr(self.Q1_head, 'detailed_forward'):
            q1, info1 = self.Q1_head.detailed_forward(phis_1)
            q2, info2 = self.Q2_head.detailed_forward(phis_2)
        else:
            q1, info1 = self.Q1_head(phis_1), None
            q2, info2 = self.Q2_head(phis_2), None

        return q1, q2, info1, info2

    def add(self, obs, action, values):
        """Method to add entries to dictionary"""
        with torch.no_grad():
            # Methods for overwriting memory
            if self.memory_overwrite == 'none':
                if self.Q1_head.mem_size >= self.Q1_head.capacity:
                    return  # Don't cycle back to overwrite previous entries
            elif self.memory_overwrite == 'cyclic':
                pass  # overwrite previous entries
            else:
                raise NotImplementedError

            # Process inputs
            inpt = obs if self.obs_type == 'pixels' else torch.cat([obs, action],
                                                                   dim=-1)
            h = self.trunk(inpt)
            h = torch.cat([h, action], dim=-1) if self.obs_type == 'pixels' else h

            phis_1 = self.Q1_neck(h)
            phis_2 = self.Q2_neck(h)

            # Add to dictionary and increment counters
            for i in range(len(obs)):
                midx_1 = self.Q1_head.mem_idx
                self.Q1_head.keys[midx_1, :] = phis_1[i, :]
                self.Q1_head.values[midx_1, :] = values[i, :]
                self.Q1_head.mem_idx = (midx_1 + 1) % self.Q1_head.capacity
                self.Q1_head.mem_size = min((self.Q1_head.mem_size + 1),
                                            self.Q1_head.capacity)

                midx_2 = self.Q2_head.mem_idx
                self.Q2_head.keys[midx_2, :] = phis_2[i, :]
                self.Q2_head.values[midx_2, :] = values[i, :]
                self.Q2_head.mem_idx = (midx_2 + 1) % self.Q2_head.capacity
                self.Q2_head.mem_size = min((self.Q2_head.mem_size + 1),
                                            self.Q2_head.capacity)
            return


class NonParamDDPGAgent(DDPGAgent):
    def __init__(self, twin_q, value_head_cfg, mc_buffer_cfg, memory_overwrite,
                 **kwargs):
        super().__init__(**kwargs)

        self.twin_q = twin_q

        self.critic = NonParametricCritic(
            self.obs_type, self.obs_dim, self.action_dim, self.feature_dim,
            self.hidden_dim, memory_overwrite,
            value_head_cfg, self.device,
        ).to(self.device)
        self.critic_target = NonParametricCritic(
            self.obs_type, self.obs_dim, self.action_dim, self.feature_dim,
            self.hidden_dim, memory_overwrite,
            value_head_cfg, self.device,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # If training only subset of parameters
        if self.grad_critic_params is not None:
            for name, param in self.critic.named_parameters():
                if not name.startswith(self.grad_critic_params):
                    param.requires_grad = False

        # optimizers
        self.encoder_opt = None
        if self.obs_type == 'pixels':
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(),
                                                lr=self.lr)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.train()
        self.critic_target.train()

        # Buffer for storing MC returns
        self.mc_buffer = utils.MCReturnBuffer(**mc_buffer_cfg)


    def add(self, time_step_obj, meta, step):
        """
        Method for agent to take online time-step information. Assumes this
        method is called for every online step.
        """
        # Unpack
        obs, action, reward = (time_step_obj.observation, time_step_obj.action,
                               time_step_obj.reward)
        discount = time_step_obj.discount
        obs_dict = {'obs': obs, 'action': action}

        # Add experiences to MC return buffer
        if not self.mc_buffer.initialized:
            self.mc_buffer.init(obs_dict)
        self.mc_buffer.add(obs_dict, reward, step)

        # End of episode add to the critic internal buffer
        if time_step_obj.last():
            b_dict = self.mc_buffer.flush()

            tensors = utils.to_torch(
                (b_dict['obs'], b_dict['action'], b_dict['return']),
                device=self.device)
            obs_tensor, act_tensor, val_tensor = tensors

            self.critic.add(obs_tensor, act_tensor, val_tensor)
            self.critic_target.add(obs_tensor, act_tensor, val_tensor)

        pass  # TODO; return metrics?

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            if self.twin_q:
                target_V = torch.min(target_Q1, target_Q2)
            else:
                target_V = target_Q1
            target_Q = reward + (discount * target_V)

        Q1, Q2, info1, info2 = self.critic.detailed_forward(obs, action)
        if self.twin_q:
            critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
        else:
            critic_loss = F.mse_loss(Q1, target_Q)

        if self.use_tb or self.use_wandb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()
            for k in info1:
                metrics[k] = info1[k]

        # optimize critic
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()
        # import ipdb; ipdb.set_trace()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        if self.use_tb or self.use_wandb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
