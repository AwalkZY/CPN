import math
import numpy as np
import torch


def analog_normal_dist(length, mean, std):
    """
    :param length: int, the total number of elements
    :param mean: int, in size (batch_size), the mean value of distributions
    :param std: float, in size (batch_size), the std value of distributions
    :return:
    """
    mean = mean.view(-1, 1).float()
    std = std.view(-1, 1).float()
    index = torch.arange(length).view(1, -1).float().to(mean.device)
    expo = - (index - mean) ** 2 / (2.0 * std ** 2)
    coef = 1.0 / math.sqrt(2 * math.pi) / std
    result = coef * torch.exp(expo)
    return result / torch.sum(result, dim=-1, keepdim=True)


# length = 64
# mean = torch.tensor([0, 0, 0, 10])
# std = torch.tensor([2.0 / 64, 32.0 / 64, 64.0 / 64, 0.5 + 1 / 8.0])
# print(analog_normal_dist(length, mean, std))
