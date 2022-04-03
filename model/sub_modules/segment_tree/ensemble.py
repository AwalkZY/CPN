import torch
from torch import nn

from model.sub_modules.router import NormalDualRouter, NormalSingleRouter
from model.sub_modules.segment_tree.base import BaseSegmentTree, get_vertical_pos, _is_branch

"""
The target of Ensemble Segment Tree is to set up a committee of leaves to make ONE decision jointly.
There will be some extra operations on single leaves.
"""


class LeafPredictor(nn.Module):
    def __init__(self, hidden_dim, class_num):
        super().__init__()
        self.core = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, class_num)
        )

    def forward(self, elements):
        # elements in (bs, leaf_num, hidden_dim) -> (bs, leaf_num, class_num)
        return self.core(elements)


class EnsembleSegmentTree(BaseSegmentTree):
    def __init__(self, model_dim: int, max_element_num: int, use_mounted_feature: bool, use_root_feature: bool,
                 hide_ratio: float, compress_method: str, attn_size: int, task_num: int,
                 use_positional_encoding: bool, class_num: int, dropout: float, graph_config: dict):
        super().__init__(model_dim, max_element_num, use_mounted_feature, use_root_feature,
                         compress_method, attn_size, use_positional_encoding, graph_config)
        self.task_num = task_num
        self.max_element_num = max_element_num
        self.branch_route = NormalSingleRouter(model_dim, task_num, self.depth + 1, dropout)
        # routes are only equipped for internal nodes
        self.leaf_predictor = LeafPredictor(model_dim, class_num)

    def calc_routing_logit(self, node_feat, mounted_feature):
        """Checked √"""
        batch_size = node_feat.size(0)
        node_depth = get_vertical_pos(torch.arange(self.node_num + 1)).float().to(node_feat.device)
        node_prob = torch.zeros(batch_size, self.node_num + 1, 1).to(node_feat.device)
        layer_prob = torch.zeros(batch_size, self.node_num + 1, 1).to(node_feat.device)
        for depth_idx in range(self.depth + 1):
            child_range = torch.arange(2 ** depth_idx, 2 ** (depth_idx + 1)).to(node_feat.device)
            parent_range = child_range // 2
            logit = self.branch_route(node_feat[:, child_range], depth_idx)
            layer_prob[:, child_range] = torch.sigmoid(logit)
            node_prob[:, child_range] += node_prob[:, parent_range] / 4.0 + layer_prob[:, child_range] / 2.0
        node_prob += (0.25 ** node_depth).view(1, -1, 1) / 2.0
        return node_prob

    def forward(self, elements, mounted_feature=None):
        final_feats = super().forward(elements)
        # final_feats in (batch_size, total_num, hidden_dim)
        routing_logits = self.calc_routing_logit(final_feats, mounted_feature)
        leaf_probs = routing_logits[:, self.leaf_num:]  # in (bs, max_len, 1)
        prediction_result = self.leaf_predictor(final_feats[:, self.leaf_num:])  # (bs, max_len, class_num)
        ensemble_result = (leaf_probs * prediction_result).sum(dim=1)  # (bs, class_num)
        return ensemble_result, prediction_result
