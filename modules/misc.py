import torch
from torch import nn


# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(data)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        """Modified cost for logarithmic updates"""
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        """Returns the matrix of $|x_i-y_j|^p$."""
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        """Barycenter subroutine, used by kinetic acceleration through extrapolation."""
        return tau * u + (1 - tau) * u1


"""NetVLAD implementation
"""
# Copyright 2018 Antoine Miech All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain data copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class NetVLAD(nn.Module):
    def __init__(self, cluster_size: object, feature_size: object, add_batch_norm: object = True) -> object:
        super().__init__()

        self.feature_size = feature_size
        self.cluster_size = cluster_size
        init_sc = (1 / math.sqrt(feature_size))
        # The `clusters` weights are the `(w,b)` in the paper
        self.clusters = nn.Parameter(init_sc * th.randn(feature_size, cluster_size))
        # The `clusters2` weights are the visual words `c_k` in the paper
        self.clusters2 = nn.Parameter(init_sc * th.randn(1, feature_size, cluster_size))
        self.add_batch_norm = add_batch_norm
        self.batch_norm = nn.BatchNorm1d(cluster_size)
        self.out_dim = cluster_size * feature_size

    def reset_parameters(self):
        # init_sc = (1 / math.sqrt(self.feature_size))
        # The `clusters` weights are the `(w,b)` in the paper
        # self.clusters = nn.Parameter(init_sc * th.randn(self.feature_size, self.cluster_size))
        # # The `clusters2` weights are the visual words `c_k` in the paper
        # self.clusters2 = nn.Parameter(init_sc * th.randn(1, self.feature_size, self.cluster_size))
        self.batch_norm.reset_parameters()

    def forward(self, x, x_mask=None, flatten=True):
        """Aggregates feature maps into a fixed size representation.  In the following
        notation, B = batch_size, N = num_features, K = num_clusters, D = feature_size.

        Args:
            x (th.Tensor): B x N x D
            x_mask: B x N
            flatten: boolean
        Returns:
            (th.Tensor): B x DK
        """
        # self.sanity_checks(x)
        max_sample = x.size()[1]
        # print(x.size(), self.feature_size)
        x = x.contiguous().view(-1, self.feature_size)  # B x N x D -> BN x D
        # if x.device != self.clusters.device:
        #      import ipdb; ipdb.set_trace()
        assignment = th.matmul(x, self.clusters)  # (BN x D) x (D x K) -> BN x K

        if self.add_batch_norm:
            assignment = self.batch_norm(assignment)

        # if x_mask is not None:
        #     assignment = assignment.masked_fill(x_mask.unsqueeze(-1) == 0, -1e30)
        assignment = F.softmax(assignment, dim=1)  # BN x K -> BN x K
        assignment = assignment.view(-1, max_sample, self.cluster_size)  # -> B x N x K
        if x_mask is not None:
            assignment = assignment.masked_fill(x_mask.unsqueeze(-1) == 0, 0)
        # assert not th.isnan(assignment).any()
        a_sum = th.sum(assignment, dim=1, keepdim=True)  # B x N x K -> B x 1 x K
        a = a_sum * self.clusters2  # B x D x K

        assignment = assignment.transpose(1, 2)  # B x N x K -> B x K x N

        x = x.contiguous().view(-1, max_sample, self.feature_size)  # BN x D -> B x N x D
        vlad = th.matmul(assignment, x)  # (B x K x N) x (B x N x D) -> B x K x D
        vlad = vlad.transpose(1, 2)  # -> B x D x K
        vlad = vlad - a

        # L2 intra norm
        vlad = F.normalize(vlad)

        # flattening + L2 norm
        if flatten:
            vlad = vlad.contiguous().view(-1, self.cluster_size * self.feature_size)  # -> B x DK
            vlad = F.normalize(vlad)
        else:
            # vlad = vlad.transpose(-1, -2) / x_mask.sum(dim=-1).unsqueeze(-1).unsqueeze(-1).float()
            vlad = vlad.transpose(-1, -2)
        return vlad  # B x DK or B x K x D

    # def sanity_checks(self, x):
    #     """Catch any nans in the inputs/clusters"""
    #     if th.isnan(th.sum(x)):
    #         print("nan inputs")
    #         ipdb.set_trace()
    #     if th.isnan(self.clusters[0][0]):
    #         print("nan clusters")
    #         ipdb.set_trace()
