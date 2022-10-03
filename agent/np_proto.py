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
from agent.np_proto_modules import *


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


class NonParametricProjectedActor(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim,
                 key_dim, predictor_grad, snd_kwargs, device):
        super().__init__()

        self.key_dim = key_dim

        # TODO: add everything to device here???
        self.predictor = nn.Linear(obs_dim, key_dim)  # .to(self.device) ??
        for p in self.predictor.parameters():
            p.requires_grad = predictor_grad

        snd_kwargs.value_dim = action_dim
        self.policy_head = SoftNeuralDictionary(**snd_kwargs).to(device)

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        dist, __ = self.detailed_forward(obs, std)
        return dist

    def detailed_forward(self, obs, std):
        h = self.predictor(obs)
        mu, info = self.policy_head.detailed_forward(h)

        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist, info


class NonParametricProtoCritic(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim,
                 pred_dim, key_dim, predictor_grad,
                 snd_kwargs, device):
        super().__init__()

        self.key_dim = key_dim

        # TODO: add everything to device here???
        self.predictor = nn.Linear(obs_dim, key_dim)  # .to(self.device) ??
        for p in self.predictor.parameters():
            p.requires_grad = predictor_grad

        self.trunk = nn.Sequential(
            nn.Linear(pred_dim + action_dim, key_dim), nn.ReLU(),
            nn.Linear(key_dim, key_dim))

        self.critic_head = SoftNeuralDictionary(**snd_kwargs).to(device)

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

        qs, info = self.critic_head.detailed_forward(h)   # qs: (B, 1)

        return qs, qs, info, None  # TODO: very hacky right now, implement twin Q later


class NonParamValueProtoAgent(ProtoAgent):
    def __init__(self, twin_q, actor_cfg, critic_cfg,
                 **kwargs):
        super().__init__(**kwargs)

        self.twin_q = twin_q
        self.protos = None  # Set to None so there is no confusion

        # Initialize actor
        actor_cfg.obs_type = self.obs_type
        actor_cfg.obs_dim = self.obs_dim
        actor_cfg.action_dim = self.action_dim
        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)

        # Initialize critic
        critic_cfg.obs_type = self.obs_type
        critic_cfg.obs_dim = self.obs_dim
        critic_cfg.action_dim = self.action_dim
        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        if self.grad_critic_params is not None:
            for name, param in self.critic.named_parameters():
                if not name.startswith(self.grad_critic_params):
                    param.requires_grad = False

        # Optimizers
        self.proto_opt = hydra.utils.instantiate(self.base_optim_cfg,
            params=utils.chain(
                self.encoder.parameters(), self.predictor.parameters(),
                self.projector.parameters())
        )  # copied directly from proto.py, while deleting self.protos

        self.actor_opt = hydra.utils.instantiate(self.actor_optim_cfg,
            params=self.actor.parameters())
        self.critic_opt = hydra.utils.instantiate(self.critic_optim_cfg,
            params=self.critic.parameters())

        self.train()
        # TODO: why not critic.train() and only critic_target.train()????
        self.critic_target.train()

    def init_from(self, other):
        # Regular initialization similar to base proto
        utils.hard_update_params(other.encoder, self.encoder)

        utils.hard_update_params(other.predictor, self.predictor)
        utils.hard_update_params(other.projector, self.projector)

        if self.init_actor:
            if type(self.actor) == NonParametricProjectedActor:
                self.actor.predictor.weight.data.copy_(other.predictor.weight.data)
                self.actor.policy_head.keys.data.copy_(other.protos.weight.data)
            elif type(self.actor) == NonParametricProtoActor:
                self.actor.policy_head.keys.data.copy_(other.protos.weight.data)
            else:
                utils.hard_update_params(other.actor, self.actor)

        if self.init_critic:
            if type(self.critic) == NonParametricProtoCritic:
                self.critic.predictor.weight.data.copy_(other.predictor.weight.data)
                self.critic.critic_head.keys.data.copy_(other.protos.weight.data)
            elif type(self.critic) == ParametricProjectedCritic:  # ablation
                self.critic.predictor.weight.data.copy_(other.predictor.weight.data)
            else:
                if self.init_critic_mode == 'only_trunk':
                    utils.hard_update_params(other.critic.trunk, self.critic.trunk)
                else:
                    raise NotImplementedError
                self.critic_target.load_state_dict(self.critic.state_dict())

    def optional_inits(self):
        """
        Method to add more initialization options, typically for ablations
        """
        if self.optional_inits_args is None:
            return

        pass

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
            if info1 is not None:
                for k in info1:
                    metrics[f'critic_{k}'] = info1[k]

        # optimize critic
        # maybe TODO: separate encoder and critic updates?
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
        dist, info = self.actor.detailed_forward(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # Get actor gradient norm if available
        # https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
        actor_grads = [torch.norm(p.grad.detach(), 2)
                       for p in self.actor.parameters() if p.grad is not None]
        actor_gradnorm = torch.norm(torch.stack(actor_grads), 2)


        if self.use_tb or self.use_wandb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
            metrics['actor_grad_norm'] = actor_gradnorm.item()

            if info is not None:
                for k in info:
                    metrics[f'actor_{k}'] = info[k]

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

        if self.use_tb or self.use_wandb:
            metrics['batch_reward'] = reward.mean().item()

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        # Update critic
        for __ in range(self.num_critic_updates):
            metrics.update(
                self.update_critic(obs, action, reward, discount,
                                   next_obs, step))

        # Update actor
        for __ in range(self.num_actor_updates):
            metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
