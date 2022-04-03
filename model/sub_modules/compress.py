import torch
import torch.nn.functional as F
from torch import nn

from modules.gates import CrossGate


class CrossGateCompress(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.cross_gate = CrossGate(hidden_dim)
        self.fuse_linear = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, elements):
        # elements in shape (batch_size, seq_len, hidden_dim)
        left_elements, right_elements = elements[:, 0::2], elements[:, 1::2]
        # in (bs, seq_len // 2, hidden_dim) (bs, seq_len // 2, hidden_dim)
        return self.fuse_linear(self.cross_gate(left_elements, right_elements))


class Compress(nn.Module):
    def __init__(self, hidden_dim, compress_ratio, method):
        super().__init__()
        assert method in ["Conv", "Pool", "CrossGate"], "Invalid Compression Prototype!"
        self.method = method
        if method == "Conv":
            self.need_transpose = True
            self.need_activation = True
            self.compress = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=compress_ratio,
                                      stride=compress_ratio)
        elif method == "Pool":
            self.need_transpose = True
            self.need_activation = False
            self.compress = nn.MaxPool1d(kernel_size=compress_ratio, stride=compress_ratio)
        elif method == "CrossGate":
            self.need_transpose = False
            self.need_activation = True
            self.compress = CrossGateCompress(hidden_dim)

    def forward(self, x):
        """
        Compress a longer sequence to a more compact one.
        The Transpose Switch should be Implemented in this module.
        :param x: in size (bs, seg_len, dim)
        :return: in size (bs, seg_len / compress_ratio, dim)
        """
        if self.need_transpose:
            result = self.compress(x.transpose(-2, -1)).transpose(-2, -1)
        else:
            result = self.compress(x)
        if self.need_activation:
            return F.gelu(result)
        else:
            return result
