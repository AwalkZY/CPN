from torch import nn
import torch


class OrderPredictor(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.disc_linear = nn.ModuleList([
            nn.Linear(2 * model_dim, model_dim) for _ in range(3)
        ])
        self.final_linear = nn.Linear(3 * model_dim, 6)

    def forward(self, features):
        # features in (batch_size * sampled_num, 3, dim)
        res_01 = self.disc_linear[0](torch.cat((features[:, 0], features[:, 1]), dim=-1))
        res_02 = self.disc_linear[1](torch.cat((features[:, 0], features[:, 2]), dim=-1))
        res_12 = self.disc_linear[2](torch.cat((features[:, 1], features[:, 2]), dim=-1))
        res = torch.cat((res_01, res_02, res_12), dim=-1)
        return self.final_linear(res)  # in (bs * sampled_num, 6)


def order2target(order):
    return {'012': 0, '021': 1, '102': 2, '120': 3, '201': 4, '210': 5}[str(order[0]) + str(order[1]) + str(order[2])]