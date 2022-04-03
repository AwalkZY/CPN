import sys

from model.sub_modules.auxiliary import order2target
from model.sub_modules.router import NormalDualRouter, NormalSingleRouter
from model.sub_modules.segment_tree.base import BaseSegmentTree, get_vertical_pos, _is_branch
from torch import nn
import torch
import numpy as np
import torch.nn.functional as F

"""
The target of Competitive Segment Tree is to pick some leaves out or get information of leaves. 
There's NO extra operation on single leaves.
"""


class CompetitiveSegmentTree(BaseSegmentTree):
    def __init__(self, model_dim: int, max_element_num: int, use_mounted_feature: bool, use_root_feature: bool,
                 compress_method: str, task_num: int, attn_size: int, use_positional_encoding: bool,
                 hide_ratio: float, dropout: float, graph_config: dict):
        super().__init__(model_dim, max_element_num, use_mounted_feature, use_root_feature, compress_method,
                         attn_size, use_positional_encoding, graph_config)
        self.hide_ratio = hide_ratio
        self.branch_route = NormalDualRouter(model_dim, task_num, self.depth + 1, dropout)
        # FIXME
        # self.branch_route = NormalSingleRouter(model_dim, task_num, self.depth + 1, dropout)

    def forward(self, elements, mounted_feature=None):
        final_feat = super().forward(elements, mounted_feature)
        boundary_prob, inside_prob = self.calc_routing_logit(final_feat)
        # FIXME
        # boundary_prob, inside_prob = self.calc_naive_logit(final_feat)
        return boundary_prob, inside_prob

    def calc_node_logit_dfs(self, node_feat, task_range):
        """Checked √"""
        batch_size = node_feat.size(0)
        node_logit = torch.zeros(batch_size, self.node_num + 1, len(task_range)).to(node_feat.device)

        def inner_loop(root_idx):
            depth = get_vertical_pos(root_idx)
            left_idx, right_idx = root_idx << 1, root_idx << 1 | 1
            root_feat, left_feat, right_feat = node_feat[:, root_idx], node_feat[:, left_idx], node_feat[:, right_idx]
            left_logit = self.branch_route(torch.cat((root_feat, left_feat), dim=-1), depth)[:, task_range]
            right_logit = self.branch_route(torch.cat((root_feat, right_feat), dim=-1), depth)[:, task_range]
            max_logit = torch.max(left_logit, right_logit)
            left_prob, right_prob = torch.exp(left_logit - max_logit), torch.exp(right_logit - max_logit)
            left_prob, right_prob = left_prob / (left_prob + right_prob), right_prob / (left_prob + right_prob)
            # left_logit = self.branch_route(torch.cat((root_feat, left_feat, right_feat),
            # dim=-1), depth)[:, task_range]
            # left_prob, right_prob = torch.sigmoid(left_logit), 1 - torch.sigmoid(left_logit)
            # all node index will keep the same, and all leaf index will become the relative order
            node_logit[:, left_idx] += node_logit[:, root_idx] + torch.log(left_prob + 1e-9)
            node_logit[:, right_idx] += node_logit[:, root_idx] + torch.log(right_prob + 1e-9)

            if _is_branch(self, left_idx):
                inner_loop(left_idx)  # calculate the left branch recursively
            if _is_branch(self, right_idx):
                inner_loop(right_idx)  # calculate the right branch recursively

        inner_loop(1)
        return node_logit

    def calc_node_prob_lfs(self, node_feat, task_range):
        """Checked √"""
        batch_size = node_feat.size(0)
        node_depth = get_vertical_pos(torch.arange(self.node_num + 1)).float().to(node_feat.device)
        node_prob = torch.zeros(batch_size, self.node_num + 1, len(task_range)).to(node_feat.device)
        layer_prob = torch.zeros(batch_size, self.node_num + 1, len(task_range)).to(node_feat.device)
        for depth_idx in range(self.depth + 1):
            child_range = torch.arange(2 ** depth_idx, 2 ** (depth_idx + 1)).to(node_feat.device)
            parent_range = child_range // 2
            logit = self.branch_route(torch.cat((node_feat[:, parent_range],
                                                 node_feat[:, child_range]), dim=-1)
                                      , depth_idx)[:, :, task_range]
            layer_prob[:, child_range] = torch.sigmoid(logit)
            node_prob[:, child_range] += node_prob[:, parent_range] / 4.0 + layer_prob[:, child_range] / 2.0
        node_prob += (0.25 ** node_depth).view(1, -1, 1) / 2.0
        return node_prob

    def naive_dfs_logit(self, node_feat):
        leaf_range = torch.arange(self.leaf_num, self.leaf_num * 2)
        logit = self.branch_route(node_feat[:, leaf_range], 0)[:, :, :2].log_softmax(1)
        # (bs, nn, dim) -> (bs, nn, tn)
        return logit

    def naive_lfs_prob(self, node_feat):
        leaf_range = torch.arange(self.leaf_num, self.leaf_num * 2)
        prob = self.branch_route(node_feat[:, leaf_range], 0)[:, :, 2:].sigmoid()
        # (bs, nn, dim) -> (bs, nn, tn)
        return prob

    # def aux_order_prediction(self, node_feat):
    #     sampled_features = []
    #     sampled_target = []
    #     batch_size, _, hidden_dim = node_feat.size()
    #     for depth in range(2, self.depth):
    #         shuffled_idx = torch.from_numpy(np.random.permutation(np.arange(2 ** depth))).long()
    #         start_pos = torch.from_numpy(np.random.permutation(np.arange(2 ** depth - 2))).long()
    #         sampled_num = 2 ** (depth - 2)
    #         for sampled_idx in range(sampled_num):
    #             chosen_range = torch.arange(start_pos[sampled_idx].item(), (start_pos[sampled_idx] + 3).item())
    #             chosen_idx = (2 ** depth) + shuffled_idx[chosen_range]
    #             batch_idx = torch.arange(batch_size)
    #             sampled_features.append(node_feat[batch_idx.view(-1, 1), chosen_idx.view(1, -1), :])
    #             # in (bs, 3, feature_dim)
    #             sampled_target.append(order2target(chosen_idx.argsort().numpy()))  # just an int
    #     sampled_features = torch.stack(sampled_features, dim=1).view(-1, 3, hidden_dim)
    #     # in (bs * sampled_num, 3, feature_dim)
    #     sampled_target = torch.tensor(sampled_target).repeat(batch_size).to(sampled_features.device)
    #     # in (bs * sampled_num)
    #     sampled_pred = self.order_predictor(sampled_features)
    #     return sampled_pred, sampled_target

    def calc_routing_logit(self, node_feat):
        # boundary_logit = self.naive_node_logit(node_feat)
        boundary_logit = self.calc_node_logit_dfs(node_feat, torch.tensor([0, 1]))[:, self.leaf_num:]
        inside_prob = self.calc_node_prob_lfs(node_feat, torch.tensor([2]))[:, self.leaf_num:]
        return boundary_logit.exp().squeeze(-1), inside_prob.squeeze(-1)

    def calc_naive_logit(self, node_feat):
        boundary_logit = self.naive_dfs_logit(node_feat)
        inside_prob = self.naive_lfs_prob(node_feat)
        return boundary_logit.exp().squeeze(-1), inside_prob.squeeze(-1)

