import torch
from torch import nn


class IndividualRouter(nn.Module):
    def __init__(self, hidden_dim, task_num, layer_num, dropout):
        super().__init__()
        self.task_num = task_num
        self.core = nn.ModuleList([nn.ModuleList([nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(p=dropout, inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(p=dropout, inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(p=dropout, inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim // 2), nn.GELU(), nn.Dropout(p=dropout, inplace=True),
            nn.Linear(hidden_dim // 2, 1)
        ) for _ in range(layer_num)]) for _ in range(task_num)])

    def forward(self, element, layer_idx):
        task_result = []
        for task_idx in range(self.task_num):
            task_result.append(self.core[task_idx][layer_idx](element).squeeze(-1))
        return torch.stack(task_result, dim=-1)


class NormalSingleRouter(nn.Module):
    def __init__(self, hidden_dim, task_num, layer_num, dropout):
        super().__init__()
        self.core = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(p=dropout, inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(p=dropout, inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim // 2), nn.GELU(), nn.Dropout(p=dropout, inplace=True),
            nn.Linear(hidden_dim // 2, task_num)
        ) for _ in range(layer_num)])

    def forward(self, element, layer_idx):
        return self.core[layer_idx](element)


class NormalDualRouter(nn.Module):
    def __init__(self, hidden_dim, task_num, layer_num, dropout):
        super().__init__()
        self.core = nn.ModuleList([nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim), nn.GELU(), nn.Dropout(p=dropout, inplace=True),
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(p=dropout, inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(p=dropout, inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(p=dropout, inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim // 2), nn.GELU(), nn.Dropout(p=dropout, inplace=True),
            nn.Linear(hidden_dim // 2, task_num)
        ) for _ in range(layer_num)])

    def forward(self, element, layer_idx):
        return self.core[layer_idx](element)
