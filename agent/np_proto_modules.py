# ==========
# Modules the np_proto class of models, including ablations
# ==========
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


def torch_entropy(X):
    """
    Compute entropy of a vector
        H(X) = - sum_{x in X} p(x) log p(x)

    If the input is a matrix, entropy of each row vector is computed
    """
    if X.dim() > 1:
        prod = X * torch.log(X)  # n x d
        ent = -prod.sum(dim=1)
    else:
        ent = -torch.dot(X, torch.log(X))
    return ent


class SoftNeuralDictionary(nn.Module):
    def __init__(self, capacity, key_dim, value_dim, temperature,
                 keys_grad, values_grad, temperature_grad, values_init,
                 score_fn_cfg, device):
        super().__init__()

        self.device = device

        self.keys = nn.parameter.Parameter(
            torch.empty((capacity, key_dim)),
            requires_grad=keys_grad)  # (M, key_dim)
        self.values = nn.parameter.Parameter(
            torch.zeros(capacity, value_dim),
            requires_grad=values_grad)  # (M, value_dim)

        self.temperature = temperature
        self.log_temp = nn.parameter.Parameter(
            torch.log(torch.tensor(temperature)),
            requires_grad=temperature_grad)

        self.score_fn = hydra.utils.instantiate(score_fn_cfg).to(device)

        # Initialize keys to be orthonormal matrix
        # TODO: make customizable, or, init to unit vectors randomly sampled in
        #       d-dimensional space
        key_init_scheme = 'orthonormal'
        if key_init_scheme == 'orthonormal':
            nn.init.orthogonal_(self.keys)
            C = self.keys.data.clone()
            C = F.normalize(C, dim=1, p=2)
            self.keys.data.copy_(C)

        # Custom initialization of values
        self.values_init = values_init
        if isinstance(values_init, float):
            nn.init.constant_(self.values, values_init)


    def forward(self, x):
        vs, info = self.detailed_forward(x)
        return vs

    def detailed_forward(self, x):
        """
        :param x: input, shape (B, key_dim)
        :return:
        """
        # Compute similarity and softmax
        scores = self.score_fn(x, self.keys)  # (B, M)
        ws = scores / torch.exp(self.log_temp)
        log_weights = ws - torch.logsumexp(ws, axis=1, keepdim=True)  # (B, M)
        weights = torch.exp(log_weights)

        # Combine with value entries to get state-action value estimates
        vs = torch.matmul(weights, self.values)  # (B, val_dim)

        with torch.no_grad():
            # Measure things
            ent_avg_batch = torch_entropy(weights.mean(dim=0)).item()
            ent_item_batch_avg = torch_entropy(weights).mean().item()

            onehot_maxw = F.one_hot(weights.max(dim=1).indices,
                                    num_classes=weights.size(1))
            ncol_max_in_batch = torch.sum(onehot_maxw.sum(dim=0) >= 1.)

            info = {
                'simscore_avg': scores.mean().item(),
                'simscore_max': scores.max(dim=1).values.mean().item(),
                'weights_avg': weights.mean().item(),
                'weights_max': weights.max(dim=1).values.mean().item(),
                'values_param_avg': self.values.mean().item(),
                'values_param_min': self.values.min().item(),
                'values_param_max': self.values.max().item(),
                'entropy_avg_batch': ent_avg_batch,
                'entropy_item_batch_avg': ent_item_batch_avg,
                'prop_col_max_in_batch': (ncol_max_in_batch/weights.size(0)).item(),
                'temperature': torch.exp(self.log_temp).item(),
            }

        return vs, info


class StochasticNeuralDictionary(SoftNeuralDictionary):
    """
    Neural dictionary where the values are sampled based on a multinomail
    distribution over key activations
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Custom initialization of values
        if isinstance(self.values_init, float):
            nn.init.constant_(self.values, self.values_init)
        # Maybe TODO: add below, but also make it possible for actor to init this
        #elif self.values_init == 'trunc_normal':
        #    nn.init.constant_(self.values, means=0.0, std=1.0, a=-0.5, b=0.5)

    def detailed_forward(self, x):
        """
        """
        # Compute similarity and softmax
        scores = self.score_fn(x, self.keys)  # (B, M)
        ws = scores / torch.exp(self.log_temp)
        log_weights = ws - torch.logsumexp(ws, axis=1, keepdim=True)  # (B, M)
        weights = torch.exp(log_weights)

        sampl_idxs = torch.multinomial(weights, num_samples=1)  # (B, 1)
        sampl_idxs = sampl_idxs.flatten()  # (B, ), assume only one sample

        # Sample
        vs = self.values[sampl_idxs]  # (B, val_dim)

        with torch.no_grad():
            # Measure things
            ent_avg_batch = torch_entropy(weights.mean(dim=0)).item()
            ent_item_batch_avg = torch_entropy(weights).mean().item()

            onehot_maxw = F.one_hot(weights.max(dim=1).indices,
                                    num_classes=weights.size(1))
            ncol_max_in_batch = torch.sum(onehot_maxw.sum(dim=0) >= 1.)

            info = {
                'simscore_avg': scores.mean().item(),
                'simscore_max': scores.max(dim=1).values.mean().item(),
                'weights_avg': weights.mean().item(),
                'weights_max': weights.max(dim=1).values.mean().item(),
                'values_param_avg': self.values.mean().item(),
                'values_param_min': self.values.min().item(),
                'values_param_max': self.values.max().item(),
                'entropy_avg_batch': ent_avg_batch,
                'entropy_item_batch_avg': ent_item_batch_avg,
                'prop_col_max_in_batch': (ncol_max_in_batch/weights.size(0)).item(),
                'temperature': torch.exp(self.log_temp).item(),
            }

        return vs, info


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
        dist, info = self.detailed_forward(obs,std)
        return dist

    def detailed_forward(self, obs, std):
        h = self.trunk(obs)
        phi = self.policy_neck(h)
        mu, info = self.policy_head.detailed_forward(phi)

        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist, info


class NonParametricProjectedStochasticActor(nn.Module):
    """
    Identical to NonParametricProjectedActor, but uses
    StochasticNeuralDictionary
    """
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim,
                 key_dim, predictor_grad, snd_kwargs, device):
        super().__init__()

        self.key_dim = key_dim

        # TODO: add everything to device here???
        self.predictor = nn.Linear(obs_dim, key_dim)  # .to(self.device) ??
        for p in self.predictor.parameters():
            p.requires_grad = predictor_grad

        snd_kwargs.value_dim = action_dim

        self.policy_head = StochasticNeuralDictionary(**snd_kwargs).to(device)

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
