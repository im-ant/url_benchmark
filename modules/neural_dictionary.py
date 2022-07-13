# ==========
# Class for neural k-nearest neighbours implementations
# ==========

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseScoreFunction(nn.Module):
    """Base class for torch score functions"""

    def __init__(self):
        super(BaseScoreFunction, self).__init__()

    def forward(self, mat1, mat2):
        """
        Compute the similarities between entries of mat1 and mat2

        :param mat1: matrix of size (n, d)
        :param mat2: matrix of size (m, d)
        :return: matrix of size (n, m) of similarity score between mat1 & mat2
        """
        raise NotImplementedError


class CosineSimilaritySF(BaseScoreFunction):
    """Cosine similarity score function"""

    def __init__(self, feature_dims):
        super(CosineSimilaritySF, self).__init__()
        self.feature_dims = feature_dims
        self.eps = 1e-6  # numerical stability

    def forward(self, mat1, mat2):
        dotmat = torch.matmul(mat1, mat2.T)  # (n, m)

        norm1 = torch.norm(mat1, dim=1)  # (n,)
        norm2 = torch.norm(mat2, dim=1)  # (m,)
        norm_prod = torch.outer(norm1, norm2)  # (n, m)
        denom = torch.max(
            norm_prod, torch.full_like(norm_prod, self.eps)
        )  # (n, m)  norm product with epsilon stability

        cossim = dotmat / denom
        return cossim


class NeuralDictionary(nn.Module):
    def __init__(self, capacity, key_dim):
        super(NeuralDictionary, self).__init__()
        self.capacity = capacity
        self.key_dim = key_dim
        self.value_dim = 1  # TODO: maybe make this customizable?

        self.keys = nn.parameter.Parameter(
            torch.empty((capacity, self.key_dim)),
            requires_grad=True)
        self.values = nn.parameter.Parameter(
            torch.empty(capacity, self.value_dim),
            requires_grad=True)

        # TODO: initialize weights here or no?
        # nn.init.normal_(self.keys)
        # nn.init.normal_(self.values)

    def forward(self, queries):
        raise NotImplementedError


class SoftNeuralDictionary(NeuralDictionary):
    def __init__(self, capacity, key_dim, temperature, score_fn_cfg):
        super(SoftNeuralDictionary, self).__init__(capacity, key_dim)
        self.temperature = temperature

        self.log_temp = nn.parameter.Parameter(
            torch.log(torch.tensor(temperature)),
            requires_grad=False)
        nn.init.zeros_(self.keys)
        nn.init.zeros_(self.values)

        self.score_fn = hydra.utils.instantiate(score_fn_cfg)

        self.mem_idx = 0
        self.mem_size = 0

    def forward(self, queries):
        """
        :param queries:
        :return: value vector of size (B, value_dim)
        """
        vs, __, __ = self.detailed_forward(queries)
        return vs

    def detailed_forward(self, queries):
        """
        :param queries: state or minibatches of states, shape (B, key_dim)
        """
        # Similarity comparison with memories
        scores = self.score_fn(queries, self.keys[:self.mem_size])  # (B, m)

        # Turn the similarity score into a softmax selection weighting
        ws = scores / torch.exp(self.log_temp)
        log_weights = ws - torch.logsumexp(ws, axis=1, keepdim=True)  # (B, m)
        weights = torch.exp(log_weights)

        # Combine with value entries to get state-action value estimates
        vs = torch.matmul(weights, self.values[:self.mem_size])  # (B, val_dim)

        return vs, weights, scores


class NeuralKNN(NeuralDictionary):
    def __init__(self, capacity, key_dim, k_neighbours):
        super(NeuralKNN, self).__init__(capacity, key_dim)
        self.k_neighbours = k_neighbours
        self.mem_idx = 0
        self.mem_size = 0

        self.eps = 1e-8

    def forward(self, queries):
        # TODO: dummy variable, fix this and use just a single one
        return self.forward_matmul(queries)

    def forward_matmul(self, queries):
        """
        Compute k-nn value estimates, implemented from scratch
        :param queries: state or minibatches of states, shape (B, key_dim)
        :return: value vector of size (B, value_dim)
        """

        # Compute the 2-norm distance, size (B, m), m=mem_size
        dists = torch.cdist(queries, self.keys[:self.mem_size, :], p=2)

        # Construct mask matrix with non-zeros entries on top k of each row
        # https://discuss.pytorch.org/t/change-values-of-top-k-in-every-row-of-tensor/39321/2
        k = min(self.k_neighbours, self.mem_size)
        kdists, k_idxs = torch.topk(dists, k=k, dim=1,
                                    largest=False, sorted=True)  # (B, m)
        mask = torch.zeros_like(dists, requires_grad=False)
        mask[torch.arange(mask.size(0))[:, None], k_idxs] = 1.

        # Differentiable top k operation
        topk_dists = dists * mask  # (B, m)
        topk_dists_hardsmooth = topk_dists / (topk_dists + self.eps)  # 1's
        ws = topk_dists_hardsmooth / self.k_neighbours  # (B, m)

        # Average top k neighbours with weights (B, value_dim)
        outputs = torch.matmul(ws, self.values[:self.mem_size, :])
        return outputs

    def forward_topk(self, queries):
        """
        Compute k-nn value estimates, implemented from scratch
        :param queries: state or minibatches of states, shape (B, key_dim)
        :return: value vector of size (B, value_dim)
        """

        # Compute the 2-norm distance, size (B, m), m=mem_size
        dists = torch.cdist(queries, self.keys[:self.mem_size, :], p=2)

        topks = torch.topk(dists, k=self.k_neighbours, dim=1,
                           largest=False, sorted=True)
        topk_dists, topk_idxs = topks.values, topks.indices  # both (B, k)

        batch_topk_vals = []
        for b_idx in range(len(queries)):  # iterate over minibatch NOTE SLOW?!
            topk_val = torch.index_select(self.values, dim=0,
                                          index=topk_idxs[b_idx])
            batch_topk_vals.append(topk_val)
        bk_vals = torch.stack(batch_topk_vals, dim=0)  # (B, k, value_dim=1)
        output = torch.mean(bk_vals, dim=1)  # (B, value_dim)

        return output

    def forward_naive(self, queries):
        """
        Compute k-nn value estimates, implemented from scratch
        :param queries: state or minibatches of states, shape (B, key_dim)
        :return: value vector of size (B, value_dim)
        """

        # For storing the k nearest distances (weights) and values
        knn_vals = torch.full((len(queries), self.k_neighbours), float('nan'),
                              device=queries.device)  # (B, k)
        knn_weights = torch.full((len(queries), self.k_neighbours), float('nan'),
                                 device=queries.device, )  # (B, k)
        # TODO: NOTE above assumes the values will be 1-dim for now

        # Compute the 2-norm distance, size (B, m), m=mem_size
        dists = torch.cdist(queries, self.keys[:self.mem_size, :], p=2)

        # Get values for k nn neighbours of each point
        sorted_idxs = torch.argsort(dists, dim=1)  # (B, m) sorted increasing

        for b_idx in range(len(queries)):  # iterate over minibatch
            # Get the min(k,m)-nearest neighbour values
            sorted_vals = self.values[sorted_idxs[b_idx], :]  # (m, value_dim)
            cur_knn_vals = sorted_vals[:self.k_neighbours]  # (min(k,m), value_dim)

            num_nns = len(cur_knn_vals)  # in case m < k

            # Populate values
            knn_vals[b_idx, :num_nns] = torch.squeeze(cur_knn_vals)

            # Populate weights
            sorted_dists = dists[b_idx, sorted_idxs[b_idx]]  # (m,), sorted
            sorted_dists = sorted_dists[:self.k_neighbours]  # (min(k,m),)

            inv_dist_ws = 1. / (sorted_dists + 1e-3)  # (min(k,m), )
            knn_weights[b_idx, :num_nns] = inv_dist_ws

        # TODO: implement and do weighted averaging!
        output = torch.mean(knn_vals, dim=1)  # (B, 1)
        output = output.unsqueeze(1)

        return output


if __name__ == '__main__':
    # feature_dims, num_actions, k_neighbours,  memory_capacity
    model = NeuralKNN(capacity=32, key_dim=4, k_neighbours=3)
    print(model)
    for name, param in model.named_parameters():
        print(name, param.size())

    batch_query = torch.randn((2, 4))
    print(batch_query, batch_query.size())
    model.mem_size = 5

    output = model(batch_query)
    print(output)
