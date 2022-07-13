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
                 feature_dim, hidden_dim, value_head_cfg):
        super().__init__()

        self.obs_type = obs_type

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

        self.Q1_head = hydra.utils.instantiate(value_head_cfg)
        self.Q2_head = hydra.utils.instantiate(value_head_cfg)

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        inpt = obs if self.obs_type == 'pixels' else torch.cat([obs, action],
                                                               dim=-1)
        h = self.trunk(inpt)
        h = torch.cat([h, action], dim=-1) if self.obs_type == 'pixels' else h

        phis_1 = self.Q1_neck(h)
        phis_2 = self.Q2_neck(h)

        q1 = self.Q1_head(phis_1)
        q2 = self.Q1_head(phis_2)

        return q1, q2

    def add(self, obs, action, values):
        """Method to add entries to dictionary"""
        with torch.no_grad():
            # TODO: temp solution to prevent circling back
            if self.Q1_head.mem_size >= self.Q1_head.capacity:
                return

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


"""
self,

name, reward_free, obs_type, obs_shape, action_shape,
device, lr, feature_dim, hidden_dim, critic_target_tau,
num_expl_steps, update_every_steps, stddev_schedule, nstep,
batch_size, stddev_clip, init_critic, use_tb, use_wandb, update_encoder, meta_dim=0
"""

class NonParamDDPGAgent(DDPGAgent):
    def __init__(self, name, reward_free, obs_type, obs_shape, action_shape,
                 device, lr, feature_dim, hidden_dim,
                 critic_target_tau, num_expl_steps, update_every_steps,
                 stddev_schedule, nstep, batch_size, stddev_clip,
                 init_critic, double_q, value_head_cfg, mc_buffer_cfg,
                 use_tb, use_wandb, update_encoder, meta_dim=0):
        super().__init__(name, reward_free, obs_type, obs_shape, action_shape,
                         device, lr, feature_dim, hidden_dim, critic_target_tau,
                         num_expl_steps, update_every_steps, stddev_schedule,
                         nstep, batch_size, stddev_clip, init_critic, use_tb,
                         use_wandb, update_encoder, meta_dim=0)

        self.double_q = double_q
        
        self.critic = NonParametricCritic(
            obs_type, self.obs_dim, self.action_dim, feature_dim,
            hidden_dim, value_head_cfg
        ).to(device)
        self.critic_target = NonParametricCritic(
            obs_type, self.obs_dim, self.action_dim, feature_dim,
            hidden_dim, value_head_cfg
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers

        if obs_type == 'pixels':
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(),
                                                lr=lr)
        else:
            self.encoder_opt = None
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()
        self.critic_target.train()

        # Buffer for storing MC returns
        self.mc_buffer = utils.MCReturnBuffer(**mc_buffer_cfg)

    def init_from(self, other):
        # copy parameters over
        utils.hard_update_params(other.encoder, self.encoder)
        utils.hard_update_params(other.actor, self.actor)
        if self.init_critic:
            utils.hard_update_params(other.critic.trunk, self.critic.trunk)

    def get_meta_specs(self):
        return tuple()

    def init_meta(self):
        return OrderedDict()

    def update_meta(self, meta, global_step, time_step, finetune=False):
        return meta

    def act(self, obs, meta, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        h = self.encoder(obs)
        inputs = [h]
        for value in meta.values():
            value = torch.as_tensor(value, device=self.device).unsqueeze(0)
            inputs.append(value)
        inpt = torch.cat(inputs, dim=-1)
        # assert obs.shape[-1] == self.obs_shape[-1]
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(inpt, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

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

        pass  # TODO; return metrics?

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            if self.double_q:
                target_V = torch.min(target_Q1, target_Q2)
            else:
                target_V = target_Q1
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        if self.double_q:
            critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
        else:
            critic_loss = F.mse_loss(Q1, target_Q)

        if self.use_tb or self.use_wandb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize critic
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def aug_and_encode(self, obs):
        obs = self.aug(obs)
        return self.encoder(obs)

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
