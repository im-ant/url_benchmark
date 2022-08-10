# ===
#
# ===
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.neural_dictionary import NeuralDictionary


class SEKernel(nn.Module):
    """
    Squared Exponential (SE) Kernel
    """
    def __init__(self, lengthscale, train_lengthscale):
        super(SEKernel, self).__init__()

        self.lengthscale = nn.parameter.Parameter(
            torch.tensor(float(lengthscale)), requires_grad=train_lengthscale)
        # TODO: make sure it is on the right device
        # TODO: do we need a scaling factor of a?

    def forward(self, x1, x2):
        """
        :param x1: tensor of shape (n, d)
        :param x2: tensor of shape (m, d)
        :return: tensor of shape (n, m) of their covariance matrix
        """
        euDist = torch.cdist(x1, x2, p=2.0)  # (n, m)
        kMat = torch.exp(-euDist / (2 * (self.lengthscale**2)))  # (n, m)
        return kMat


class IIDGaussianNoiseGP(NeuralDictionary):
    def __init__(self, noise_var, kernel_fn_cfg,
                 **kwargs):
        super(IIDGaussianNoiseGP, self).__init__(**kwargs)

        self.kernel = hydra.utils.instantiate(kernel_fn_cfg)
        self.noise_var = noise_var  # TODO: make this trainable parameter

        self.KMat = None
        self.invKIMat = None

        # TODO: add bias, and what is the prior function?

        self.mem_idx = 0  # TODO make multi dimensional?
        self.mem_size = 0
        # TODO: pre-compute the inverse covariance

    def _precompute_covariances(self):
        """
        Helper function to optionally pre-compute the vocariance matrix
        """
        KMat = self.kernel(self.keys[:self.mem_size, :],
                           self.keys[:self.mem_size, :])  # (M, M)
        iMat = torch.eye(self.mem_size, device=KMat.device)

        invKIMat = torch.linalg.inv(KMat + (self.noise_var * iMat))

        return KMat, invKIMat

    def inplace_compute_covariances(self):
        KMat, invKIMat = self._precompute_covariances()
        self.KMat, self.invKIMat = KMat, invKIMat

    def forward(self, queries):
        """
        :param queries: state or minibatches of states, shape (B, key_dim)
        :return: mean prediction of y, shape (B, value_dim)
        """
        self.inplace_compute_covariances()  # TODO: do this every step?
        if self.invKIMat == None:
            raise Exception('inverse covariance matrix is not initialized')

        kstar = self.kernel(queries, self.keys[:self.mem_size, :])  # (B, M)
        y_mean = kstar @ self.invKIMat @ self.values[:self.mem_size,:]  # (B, ?)
        return y_mean

    def detailed_forward(self, queries):
        self.inplace_compute_covariances()  # TODO: do this every step?
        if self.invKIMat == None:
            raise Exception('inverse covariance matrix is not initialized')

        kstar = self.kernel(queries, self.keys[:self.mem_size, :])  # (B, M)
        kInvKIMat = kstar @ self.invKIMat  # (B, M)

        # Compute mean prediction
        y_mean = kInvKIMat @ self.values[:self.mem_size,:]  # (B, ?)

        # Compute variance of each element in batch
        bCov = self.kernel(queries.unsqueeze(1), queries.unsqueeze(1))  # (B, 1, 1)
        bxCov = (kInvKIMat.unsqueeze(1) @
                 kstar.unsqueeze(1).transpose(1,2))  # (B, 1, M) @ (B, M, 1)
        batch_var = (bCov - bxCov).squeeze()  # (B,)

        # Metrics
        cur_metrics = {
            'batch_f_mean_var': batch_var.detach().mean().item()
        }

        return y_mean, cur_metrics
