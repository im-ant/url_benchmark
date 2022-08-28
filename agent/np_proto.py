from copy import deepcopy

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch import jit

import utils
from agent.ddpg import DDPGAgent
from agent.proto import ProtoAgent


class BaseSimilarityFunction(nn.Module):
    """
    NOTE: this is not used currently
    """
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        """
        :param x1: (n, d)
        :param x2: (m, d)
        """
        return None


class NonParametricProtoCritic(nn.Module):
    def __init__(self, obs_dim, pred_dim, action_dim, capacity,
                 protos_dim, value_dim, predictor_grad, protos_grad, values_grad,
                 temperature, score_fn_cfg,
                 device):
        super().__init__()

        self.capacity = capacity
        self.protos_dim = protos_dim
        self.value_dim = value_dim

        # TODO: add everything to device here???
        self.predictor = nn.Linear(obs_dim, pred_dim)  #.to(self.device) ??
        for p in self.predictor.parameters():
            p.requires_grad = predictor_grad

        self.trunk = nn.Sequential(
            nn.Linear(pred_dim + action_dim, protos_dim), nn.ReLU(),
            nn.Linear(protos_dim, protos_dim))

        self.protos = nn.parameter.Parameter(
            torch.empty((capacity, protos_dim)),
            requires_grad=protos_grad)
        self.values = nn.parameter.Parameter(
            torch.zeros(capacity, value_dim),
            requires_grad=values_grad)

        # TODO: initialize similarity function
        self.temperature = temperature
        self.log_temp = nn.parameter.Parameter(
            torch.log(torch.tensor(temperature)),
            requires_grad=False)
        self.score_fn = hydra.utils.instantiate(score_fn_cfg)

        self.apply(utils.weight_init)  # TODO: apply better weight initialization?

    def forward(self, obs, action):
        q1, q2, info1, info2 = self.detailed_forward(obs, action)
        return q1, q2

    def detailed_forward(self, obs, action):
        # Project (named "predictor" in proto) input and normalize
        inpt = self.predictor(obs)
        inpt = F.normalize(inpt, dim=1, p=2)

        # Concat with action and project again
        h = torch.cat([inpt, action], dim=-1)
        h = self.trunk(h)  # (B, key_dim)

        # Compute similarity and softmax
        # TODO: probably move this to a separate class for computing similarity function
        scores = self.score_fn(h, self.protos)  # (B, M)
        ws = scores / torch.exp(self.log_temp)
        log_weights = ws - torch.logsumexp(ws, axis=1, keepdim=True)  # (B, M)
        weights = torch.exp(log_weights)

        # Combine with value entries to get state-action value estimates
        qs = torch.matmul(weights, self.values)  # (B, val_dim)

        with torch.no_grad():
            info = {
                'simscore_avg': scores.mean().item(),
                'simscore_max': scores.max().item(),
                'weights_avg': weights.mean().item(),
                'weights_max': weights.max().item(),
                'values_param_avg': self.values.mean().item(),
                'values_param_min': self.values.min().item(),
                'values_param_max': self.values.max().item(),
            }

        return qs, qs, info, None  # TODO: very hacky right now, implement twin Q later


class NonParamValueProtoAgent(ProtoAgent):
    def __init__(self, twin_q, critic_kwargs, **kwargs):
        super().__init__(**kwargs)

        self.twin_q = twin_q
        self.protos = None  # Set to None so there is no confusion

        critic_kwargs.obs_dim = self.obs_dim
        critic_kwargs.action_dim = self.action_dim
        self.critic = NonParametricProtoCritic(**critic_kwargs).to(self.device)
        self.critic_target = NonParametricProtoCritic(
            **critic_kwargs).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.proto_opt = torch.optim.Adam(utils.chain(
            self.encoder.parameters(), self.predictor.parameters(),
            self.projector.parameters()),
            lr=self.lr)  # TODO: name this better

        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.train()
        # TODO: why not critic.train() and only critic_target.train()????
        self.critic_target.train()

    def init_from(self, other):
        # Regular initialization similar to base proto
        utils.hard_update_params(other.encoder, self.encoder)
        if self.init_actor:
            utils.hard_update_params(other.actor, self.actor)

        utils.hard_update_params(other.predictor, self.predictor)
        utils.hard_update_params(other.projector, self.projector)

        if self.init_critic:
            # Initialize from a pre-trained proto agent
            if type(other) == ProtoAgent:
                self.critic.predictor.weight.data.copy_(other.predictor.weight.data)
                self.critic.protos.data.copy_(other.protos.weight.data)
            # ??
            else:
                raise NotImplementedError

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
        #import ipdb; ipdb.set_trace()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)

        if self.use_tb or self.use_wandb:
            metrics['batch_reward'] = reward.mean().item()

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        # Update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # Update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
