from collections import OrderedDict

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class ConvEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        assert len(obs_shape) == 3
        self.conv_out_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU())

        self.fc = nn.Sequential(nn.Linear(self.conv_out_dim, feature_dim),
                                nn.LayerNorm(feature_dim), nn.Tanh())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        h = self.fc(h)
        return h


def mlp(input_dim, hidden_dim, hidden_depth, output_dim, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


class DeepCritic(nn.Module):
    """
    NOTE: ddpg.py's critic has the structures:
            pixel: obs -> [feature -> hidden -> hidden -> 1]
            state: obs -> [hidden -> hidden -> 1]
        where square bracket denote the double-q learning components.
        Instead, for DQN we will just have the
                [hidden -> hidden -> |A|] structure
        and move the first feature layer into the convolutional encoder.
        Note the layer-norm structure is also a bit different.
    """

    def __init__(self, input_dim, hidden_dim, hidden_depth, num_actions,
                 dueling):
        super().__init__()

        self.dueling = dueling

        if self.dueling:
            self.V = mlp(input_dim, hidden_dim, hidden_depth, 1)
            self.A = mlp(input_dim, hidden_dim, hidden_depth, num_actions)
        else:
            self.Q = mlp(input_dim, hidden_dim, hidden_depth, num_actions)

        self.apply(utils.weight_init)  # TODO: following ddpg, not sure if required?

    def forward(self, obs):
        if self.dueling:
            v = self.V(obs)
            a = self.A(obs)
            q = v + a - a.mean(1, keepdim=True)
        else:
            q = self.Q(obs)
        return q


class BaseValueBasedAgent:
    """Value based agent"""

    def __init__(self, name, reward_free, obs_type, obs_shape,
                 init_critic, update_encoder, use_tb, use_wandb,
                 device, meta_dim=0):
        self.reward_free = reward_free
        self.obs_type = obs_type
        self.obs_shape = obs_shape
        self.init_critic = init_critic
        self.update_encoder = update_encoder

        self.use_tb = use_tb
        self.use_wandb = use_wandb

        self.device = device
        self.meta_dim = meta_dim

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        # self.actor.train(training)  # TODO fix this
        self.critic.train(training)

    def init_from(self, other):
        # TODO: directly copied from ddpg, need to maybe update this
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


class DQNAgent(BaseValueBasedAgent):
    """Deep Q Learning (with various tricks)"""

    def __init__(self, num_actions, num_expl_steps, eps_schedule,
                 nstep, batch_size, lr, double_q, max_grad_norm,
                 encoder_cfg, critic_cfg, update_every_steps,
                 critic_target_tau, critic_target_freq,
                 **kwargs):
        super().__init__(**kwargs)

        # TODO: need to clean up the arguments! also maybe move common ones to parent class

        self.num_actions = num_actions

        self.critic_target_update_frequency = None
        self.max_grad_norm = None
        self.double_q = double_q

        self.nstep = nstep
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

        self.num_expl_steps = num_expl_steps
        self.eps_schedule = eps_schedule  # TODO: testing this, fix

        self.update_every_steps = update_every_steps
        self.lr = lr  # TODO: should use a better optimizer init instead

        self.critic_target_tau = critic_target_tau
        self.critic_target_freq = critic_target_freq

        # Discretize the continuous action space  TODO moved to env, delete this
        # self.action_embedding = GridActionEmbedder(self.action_shape)

        # Initialize models
        if self.obs_type == 'pixels':
            encoder_cfg.obs_shape = self.obs_shape
            self.aug = utils.RandomShiftsAug(pad=4)
            self.encoder = hydra.utils.instantiate(encoder_cfg).to(self.device)
            self.obs_dim = self.encoder.conv_out_dim + self.meta_dim
        else:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = self.obs_shape[0] + self.meta_dim

        self.actor = None

        critic_cfg.input_dim = self.obs_dim
        critic_cfg.num_actions = self.num_actions
        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        # TODO: set independent optimizer parameters for these?
        self.encoder_opt = None
        if self.obs_type == 'pixels':
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(),
                                                lr=lr)
        self.actor_opt = None
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.training = True
        self.train()
        self.critic_target.train()

        self.update_counter = 0

    def act(self, obs, meta, step, eval_mode):
        # TODO maybe: can this whole method be in the base class instead?

        # Encode observation
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        h = self.encoder(obs)
        inputs = [h]
        for value in meta.values():
            value = torch.as_tensor(value, device=self.device).unsqueeze(0)
            inputs.append(value)
        inpt = torch.cat(inputs, dim=-1)

        # Discrete action-value outputs
        q = self.critic(inpt)

        # Sample action with eps-greedy policy
        expl_eps = utils.schedule(self.eps_schedule, step)
        sample_random_action = False
        if not eval_mode:
            if step < self.num_expl_steps:
                sample_random_action = True
            elif np.random.choice([True, False], p=[expl_eps, 1 - expl_eps]):
                sample_random_action = True

        if sample_random_action:
            action = np.random.choice(self.num_actions, size=())
        else:
            action = q.max(dim=1)[1].cpu().numpy()[0]
        return action.astype(np.int32)

    def train(self, training=True):
        self.training = training
        self.critic.train(training)

    def add(self, time_step_obj, meta, step):
        # Dummy method
        pass

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        """
        :param discount: assume it is already gamma**nstep
        """
        metrics = dict()

        # TODO: do we need the variable not_done for DM_control?
        with torch.no_grad():
            if self.double_q:
                next_Q_target = self.critic_target(next_obs)
                next_Q = self.critic(next_obs)
                next_action_idxs = next_Q.max(dim=1)[1].unsqueeze(1)
                next_Q = next_Q_target.gather(1, next_action_idxs)
                target_Q = reward + (discount * next_Q)
            else:
                next_Q = self.critic_target(next_obs)
                next_Q = next_Q.max(dim=1)[0].unsqueeze(1)
                target_Q = reward + (discount * next_Q)

        current_Q  = self.critic(obs)  # (B, num_actions)
        action_as_idxs = action.type(torch.int64).unsqueeze(1)  # (B, 1)
        current_Q = current_Q.gather(1, action_as_idxs)  # (B, 1)

        # Compute error
        td_errors = current_Q - target_Q
        critic_losses = F.smooth_l1_loss(current_Q, target_Q, reduction='none')
        critic_loss = critic_losses.mean()

        if self.use_tb or self.use_wandb:
            with torch.no_grad():
                # TODO: move this to outer loop, since there might be two critics?
                metrics['critic_target_q'] = target_Q.mean().item()
                metrics['critic_q'] = current_Q.mean().item()
                # metrics['critic_q2'] = Q2.mean().item()  # TODO fix
                metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()

        if self.max_grad_norm > 0.0:
            nn.utils.clip_grad_norm_(self.critic.parameters(),
                                     self.max_grad_norm)
            if self.encoder_opt is not None:
                nn.utils.clip_grad_norm_(self.encoder.parameters(),
                                         self.max_grad_norm)

        self.critic_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        return metrics

    def aug_and_encode(self, obs):
        obs = self.aug(obs)
        return self.encoder(obs)

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        # Sample buffer. NOTE: assume discount is actually gamma**n_step
        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # Augment and encode
        cur_s = self.aug_and_encode(obs)
        with torch.no_grad():
            next_s = self.aug_and_encode(next_obs)

        if self.use_tb or self.use_wandb:
            # TODO: why only when use_tb? can't we always have this?
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        critic_metrics = self.update_critic(cur_s, action, reward, discount,
                                            next_s, step)
        metrics.update(critic_metrics)

        # update critic target
        if self.update_counter % 1 == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_target_tau)

        self.update_counter += 1

        return metrics

    def old_get_metric_keys(self):
        """Keys to initialize the logger with"""
        # TODO: not longer useful and delete?

        keys = ['critic_loss', 'batch_reward']
        return keys
