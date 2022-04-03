from torch import nn
import torch
import torch.nn.functional as F
import math


class PositionEncoder(nn.Module):
    """Implement the PE function."""

    def __init__(self, input_dim, max_len=5000):
        super().__init__()

        # Compute the positional encodings once in log space.
        pos_encoding = torch.zeros(max_len, input_dim)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., input_dim, 2) *
                             -(math.log(10000.0) / input_dim))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        pos_encoding = pos_encoding.unsqueeze(0).clone().detach().requires_grad_(False)
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, x):
        return self.pos_encoding[:, :x.size(1)]


class ArbitraryPositionEncoder(nn.Module):
    """Implement the PE function."""

    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.div_term = torch.exp(torch.arange(0, input_dim, 2) *
                                  -(math.log(10000.0) / input_dim)).requires_grad_(False)

    def forward(self, x):
        pos_encoding = torch.zeros(x.size(0), self.input_dim).to(x.device)
        pos_encoding[:, 0::2] = torch.sin(x.unsqueeze(1) * self.div_term.clone().detach_().to(x.device))
        pos_encoding[:, 1::2] = torch.cos(x.unsqueeze(1) * self.div_term.clone().detach_().to(x.device))
        return pos_encoding
