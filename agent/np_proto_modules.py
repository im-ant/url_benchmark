# ==========
# Modules the np_proto class of models, including ablations
# ==========
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
# from agent.np_proto import SoftNeuralDictionary


class ParametricProjectedCritic(nn.Module):
    """
    Ablation model for agent.np_proto.NonParametricProtoCritic

    """
    def __init__(self, obs_type, obs_dim, action_dim,
                 pred_dim, key_dim, predictor_grad, q_hidden_dim):
        super().__init__()

        self.key_dim = key_dim

        # TODO: add everything to device here???
        self.predictor = nn.Linear(obs_dim, key_dim)  # .to(self.device) ??
        for p in self.predictor.parameters():
            p.requires_grad = predictor_grad

        self.trunk = nn.Sequential(
            nn.Linear(pred_dim + action_dim, key_dim), nn.ReLU(),
            nn.Linear(key_dim, key_dim), nn.ReLU())

        def make_q():
            q_layers = []
            q_layers += [
                nn.Linear(key_dim, q_hidden_dim), nn.ReLU(inplace=True)
            ]
            q_layers += [nn.Linear(q_hidden_dim, 1)]
            return nn.Sequential(*q_layers)

        self.critic_head = make_q()

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

        qs = self.critic_head(h)  # qs: (B, 1)

        return qs, qs, None, None  # TODO: very hacky right now, implement twin Q later


class NonParametricProtoActor(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim,
                 key_dim, snd_kwargs, device):
        super().__init__()

        # feature_dim = feature_dim if obs_type == 'pixels' else hidden_dim

        # TODO: alternatively just use a single projector to get to the actor??

        self.trunk = nn.Sequential(nn.Linear(obs_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        policy_layers = []
        policy_layers += [
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(inplace=True),
        ]
        # NOTE: not using this for now because using non-parametric instead
        # add additional hidden layer for pixels
        # if obs_type == 'pixels':
        #    policy_layers += [
        #        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)
        #    ]
        policy_layers += [nn.Linear(hidden_dim, key_dim)]
        self.policy_neck = nn.Sequential(*policy_layers)

        snd_kwargs.value_dim = action_dim
        self.policy_head = SoftNeuralDictionary(**snd_kwargs).to(device)

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)
        phi = self.policy_neck(h)
        mu = self.policy_head(phi)

        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist

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
        h = self.predictor(obs)
        mu = self.policy_head(h)

        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


