import torch
import math
from torch import nn
from torch.nn import LayerNorm
from torch_geometric import nn as graph_nn
from torch_geometric.nn import DeepGCNLayer

from model.sub_modules.compress import Compress
from modules.position_layers import ArbitraryPositionEncoder


class GraphComponent(nn.ModuleList):
    # Checked √
    def __init__(self, name, layer_num, model_dim, dropout, edge_type, feedforward_dim, gnn_config):
        super().__init__()
        self.layer_num = layer_num
        self.name = name
        self.model_dim = model_dim
        self.feedforward_dim = feedforward_dim
        self.edge_type = edge_type
        self.layers = self._construct_deep_layers(name, layer_num, dropout, gnn_config)
        # self.gnn_layers = self._construct_gnn_layers(name, layer_num, gnn_config)
        # self.zip_layers = self._construct_zip_layers(layer_num, gnn_config["heads"])
        # self.ffn_layers = self._construct_ffn_layers(layer_num)
        # self.norm_layers = self._construct_norm_layers(2 * layer_num)

    def _construct_deep_layers(self, name, layer_num, dropout, gnn_config):
        layers = []
        for layer_idx in range(layer_num):
            conv = getattr(graph_nn, name + "Conv")(**gnn_config)
            norm = LayerNorm(gnn_config["out_channels"], elementwise_affine=True)
            act = nn.GELU()
            layers.append(DeepGCNLayer(conv, norm, act, block='res+', dropout=dropout))
        return nn.ModuleList(layers)

    def forward(self, node_feat, edge_index, edge_type):
        # edge_type = self._adjust_edge_type(edge_type)
        for layer_idx in range(self.layer_num):
            # original_feat = node_feat
            # node_feat = F.dropout(F.gelu(self.gnn_layers[layer_idx](node_feat, edge_index)), 0.4)
            # node_feat = self.zip_layers[layer_idx](node_feat) + original_feat
            # node_feat = self.norm_layers[2 * layer_idx](node_feat)
            # node_feat = self.ffn_layers[layer_idx](node_feat) + node_feat
            # node_feat = self.norm_layers[2 * layer_idx + 1](node_feat)
            # node_feat = self.layers[layer_idx](node_feat, edge_index, edge_type)
            node_feat = self.layers[layer_idx](node_feat, edge_index)
        return node_feat

    # def _construct_zip_layers(self, layer_num, head_num):
    #     return nn.ModuleList([
    #         nn.Sequential(
    #             nn.Linear(self.model_dim * head_num, self.model_dim),
    #             nn.GELU(),
    #             nn.Dropout(0.2)
    #         ) for _ in range(layer_num)
    #     ])

    def _construct_norm_layers(self, layer_num):
        return nn.ModuleList([
            nn.LayerNorm(self.model_dim) for _ in range(layer_num)
        ])

    def _adjust_edge_type(self, edge_type):
        # Checked √
        if self.edge_type == "Categorical":
            return (edge_type > 0).long()  # torch.sign(edge_type) + 1
        elif self.edge_type == "Hierarchical":
            return edge_type - edge_type.min()
        else:
            raise NotImplementedError

    # def _construct_ffn_layers(self, layer_num):
    #     return nn.ModuleList([
    #         nn.Sequential(
    #             nn.Linear(self.model_dim, self.feedforward_dim),
    #             nn.GELU(),
    #             nn.Dropout(0.2),
    #             nn.Linear(self.feedforward_dim, self.model_dim),
    #             nn.GELU(),
    #             nn.Dropout(0.2)
    #         ) for _ in range(layer_num)
    #     ])

    @staticmethod
    def _construct_gnn_layers(name, layer_num, gnn_config):
        return nn.ModuleList([getattr(graph_nn, name + "Conv")(**gnn_config) for _ in range(layer_num)])


def _is_leaf(tree, node_idx):
    # Checked √
    return node_idx >= tree.leaf_num


def _is_branch(tree, node_idx):
    # Checked √
    return node_idx < tree.leaf_num


def get_vertical_pos(index):
    if isinstance(index, torch.Tensor):
        index = index.float().masked_fill(index == 0, 0.5)
        return torch.floor(torch.log2(index)).long()
    else:
        return math.floor(math.log2(1.0 * index)) if index != 0 else -1


def get_horizontal_pos(index, vertical_pos=None):
    # Checked √
    if vertical_pos is None:
        vertical_pos = get_vertical_pos(index)
    return index - 2 ** vertical_pos


def in_layer(node_idx, depth):
    return 2 ** depth <= node_idx < 2 ** (depth + 1)


class BaseSegmentTree(nn.Module):
    def __init__(self, model_dim: int, max_element_num: int, use_mounted_feature: bool, use_root_feature: bool,
                 compress_method: str, attn_size: int, use_positional_encoding: bool,
                 graph_config: dict):
        """
        It's worth noting that {Nodes} = {Branches} + {Leaves}.
        :param model_dim: the dimension of elements on the segment tree
        :param max_element_num: the maximal number of leaf nodes
        :param compress_method: the method adopted to compress sub_node information
        :param attn_size: the number of node to be attended at every level
        :param use_positional_encoding: whether to use positional encoding to enhance the performance
        :param use_mounted_feature: whether to mount another cross-modal feature to ...
        :param graph_config: the config of gnn
        """
        super().__init__()
        self.use_root_feature = use_root_feature
        self.use_mounted_feature = use_mounted_feature
        self.attn_size = attn_size
        self.use_positional_encoding = use_positional_encoding
        self.depth = math.ceil(math.log2(max_element_num))
        self.leaf_num = 2 ** self.depth
        self.branch_num = self.leaf_num - 1
        self.node_num = self.branch_num + self.leaf_num
        self.compress = self._construct_compress(model_dim, compress_method)
        self.graph_component = GraphComponent(**graph_config)
        self.vertical_encoder = ArbitraryPositionEncoder(model_dim // 2)
        self.horizontal_encoder = ArbitraryPositionEncoder(model_dim // 2)
        self.edge_index, self.edge_vectors = self._construct_graph()
        self.horizontal_encoding, self.vertical_encoding = self._calculate_positional_encoding()
        # compress_ratio should be hard-coded as 2 cause it's a segment tree

    def _fetch_edge_data(self, device, batch_size):
        edge_num = len(self.edge_index[0])
        batch_order = (torch.arange(batch_size).view(-1, 1).repeat(1, edge_num).view(-1) *
                       (self.node_num + 1)).view(1, -1).to(device)
        edge_index = torch.tensor(self.edge_index).to(device).repeat(1, batch_size) + batch_order
        edge_type = torch.tensor(self.edge_vectors).to(device).repeat(batch_size)
        return edge_index, edge_type

    def _fetch_positional_encoding(self, device, batch_size):
        horizontal_encoding = self.horizontal_encoding.repeat(batch_size, 1).to(device)
        vertical_encoding = self.vertical_encoding.repeat(batch_size, 1).to(device)
        return horizontal_encoding, vertical_encoding

    def _calculate_positional_encoding(self):
        # Checked √
        index = torch.arange(self.node_num + 1)
        vertical_pos = get_vertical_pos(index)
        horizontal_pos = get_horizontal_pos(index, vertical_pos)
        return self.horizontal_encoder(horizontal_pos), self.vertical_encoder(vertical_pos)

    def forward(self, elements, mounted_feature=None):
        """
        :param mounted_feature: mounted cross-modal feature
        :param elements: elements used to construct the tree
        :return final_feat: in (batch_size, node_num + 1, hidden_dim)
        """
        edge_index, edge_type = self._fetch_edge_data(elements.device, elements.size(0))
        node_feat = self._construct_tree(elements)
        if mounted_feature is not None and self.use_mounted_feature:
            node_feat[:, 0] = mounted_feature
        else:
            node_feat[:, 0].fill_(-1)
        horizontal_encoding, vertical_encoding = self._fetch_positional_encoding(elements.device, elements.size(0))
        concat_encoding = torch.cat((horizontal_encoding, vertical_encoding), dim=-1)
        if self.use_positional_encoding:
            flatten_node_feat = node_feat.view_as(concat_encoding) + concat_encoding
        else:
            flatten_node_feat = node_feat.view_as(concat_encoding)
        final_feat = self.graph_component.forward(flatten_node_feat, edge_index, edge_type).view_as(node_feat)
        return final_feat

    def calc_routing_logit(self, *args, **kwargs):
        raise NotImplementedError

    def _trace_children(self, node_idx: int):
        """
        checked √
        :param node_idx: the index of the root node of the subtree to be queried
        :return: a list in node index
        """
        terminal_idx, edge_vectors = [], []
        start, end = node_idx << 1, node_idx << 1 | 1
        rel_depth = 1
        while (start <= self.node_num) and (end <= self.node_num):
            terminal_idx.extend(list(range(start, end + 1)))
            edge_vectors.extend([rel_depth] * (end + 1 - start))
            start = start << 1
            end = end << 1 | 1
            rel_depth += 1
        return terminal_idx, edge_vectors

    def _trace_left(self, start_idx: int):
        """
        :param start_idx: the start of interval (order in leaves)
        :return: the same to "_trace_interval"
        """
        node_idx = start_idx + self.leaf_num
        original_depth = current_depth = self.depth
        terminal_index, edge_vectors = [], []
        stop_sign = False
        while not stop_sign:
            current_start = node_idx - self.attn_size + 1 - ((node_idx - self.attn_size + 1) & 1)
            stop_sign = current_start < 2 ** current_depth
            current_start = max(2 ** current_depth, current_start)
            current_end = node_idx
            terminal_index.extend(list(range(current_start, current_end + 1)))
            edge_vectors.extend([current_depth - original_depth] * (current_end - current_start + 1))
            node_idx = (current_start - 1) >> 1
            current_depth -= 1
        return terminal_index, edge_vectors

    def _trace_right(self, end_idx: int):
        """
        :param end_idx: the end of interval (order in leaves)
        :return: the same to "_trace_interval"
        """
        node_idx = end_idx + self.leaf_num
        original_depth = current_depth = self.depth
        terminal_index, edge_vectors = [], []
        stop_sign = False
        while not stop_sign:
            current_start = node_idx
            current_end = node_idx + self.attn_size - 1 + ((node_idx - self.attn_size) & 1)
            stop_sign = current_end >= 2 ** (current_depth + 1)
            current_end = min(2 ** (current_depth + 1) - 1, current_end)
            terminal_index.extend(list(range(current_start, current_end + 1)))
            edge_vectors.extend([current_depth - original_depth] * (current_end - current_start + 1))
            node_idx = (current_end + 1) >> 1
            current_depth -= 1
        return terminal_index, edge_vectors

    def _trace_interval(self, start_idx: int, end_idx: int):
        """
        checked √
        :param start_idx: the start of interval (order in leaves)
        :param end_idx: the end of interval (order in leaves)
        :return: a list in node index
        """
        if start_idx > end_idx:
            return [], []
        elif start_idx == end_idx:
            return [start_idx + self.leaf_num], [0]
        else:
            terminal_index, edge_vectors = [], []
            rel_depth = 0
            start = int(start_idx + self.leaf_num - 1)
            end = int(end_idx + self.leaf_num + 1)
            while start ^ end ^ 1:
                if ~start & 1:
                    terminal_index.append(start ^ 1)
                    edge_vectors.append(rel_depth)
                if end & 1:
                    terminal_index.append(end ^ 1)
                    edge_vectors.append(rel_depth)
                start >>= 1
                end >>= 1
                rel_depth -= 1
            return terminal_index, edge_vectors

    def _construct_graph(self):
        # checked √
        edge_start = []
        edge_end = []
        edge_vectors = []
        for node_idx in range(1, self.node_num + 1):
            if node_idx < self.leaf_num:  # branch node
                descendant_terminals, descendant_vectors = self._trace_children(int(node_idx))
                edge_start.extend(descendant_terminals)
                edge_end.extend([node_idx] * len(descendant_terminals))
                edge_vectors.extend(descendant_vectors)
            else:  # leaf node
                leaf_idx = node_idx - self.leaf_num
                left_terminals, left_vectors = self._trace_left(leaf_idx - 1)
                # self._trace_interval(0, leaf_idx - 1)
                right_terminals, right_vectors = self._trace_right(leaf_idx + 1)
                # self._trace_interval(leaf_idx + 1, self.leaf_num - 1)
                # Here we try to include the root node
                edge_start.extend([*left_terminals, *right_terminals])
                edge_end.extend([node_idx] * len([*left_terminals, *right_terminals]))
                edge_vectors.extend([*left_vectors, *right_vectors])
                if self.use_mounted_feature:
                    edge_start.append(0)
                    edge_end.append(node_idx)
                    edge_vectors.append(self.depth)
                if self.use_root_feature:
                    edge_start.append(1)
                    edge_end.append(node_idx)
                    edge_vectors.append(self.depth)
        return [edge_start, edge_end], edge_vectors

    def _construct_compress(self, model_dim, compress_method):
        # Checked √
        return Compress(model_dim, 2, compress_method)
        # return nn.ModuleList(Compress(model_dim, 2, compress_method) for _ in range(self.branch_num))

    def _construct_tree(self, x):
        """
        Checked √
        :param x: input sequence in shape (bs, max_len, dim), x should be padded before construction.
        :return: in shape (bs, node_num + 1, dim)
        """
        batch_size, max_len, hidden_dim = x.size()
        node_feat = torch.zeros(batch_size, self.node_num + 1, hidden_dim).to(x.device)
        crt_num = self.leaf_num  # the number of nodes at this level
        max_pos = min(2 * crt_num, crt_num + max_len)
        node_feat[:, crt_num: max_pos, :] = x
        for depth_idx in range(self.depth):
            crt_start, crt_end = crt_num, 2 * crt_num  # the start idx and end idx of node at this level
            next_start, next_end = crt_num // 2, crt_num  # the start idx and end idx of node at the upper level
            crt_feat = node_feat[:, crt_start: crt_end, :].clone()
            # node_feat[:, next_start: next_end, :] = self.compress[depth_idx](crt_feat)
            node_feat[:, next_start: next_end, :] = self.compress(crt_feat)
            crt_num //= 2
        return node_feat
