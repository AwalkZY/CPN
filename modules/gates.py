import torch
import torch.nn as nn


class CrossGate(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear_transformation1 = nn.Linear(input_dim, input_dim, bias=True)
        self.linear_transformation2 = nn.Linear(input_dim, input_dim, bias=True)

    def forward(self, x1, x2):
        g1 = torch.sigmoid(self.linear_transformation1(x1))
        h2 = g1 * x2
        g2 = torch.sigmoid(self.linear_transformation2(x2))
        h1 = g2 * x1
        return torch.cat([h1, h2], dim=-1)
        # return h1, h2
