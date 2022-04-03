import numpy as np
import math
import torch
import matplotlib.pyplot as plt


def gd(x, mu=0, sigma=1):
    """根据公式，由自变量x计算因变量的值

    Argument:
        x: array
            输入数据（自变量）
        mu: float
            均值
        sigma: float
            方差
    """
    left = 1 / (np.sqrt(2 * math.pi) * np.sqrt(sigma))
    right = np.exp(-(x - mu)**2 / (2 * sigma))
    return left * right


if __name__ == '__main__':
    #  自变量
    np.random.seed(100)
    torch.manual_seed(100)
    x = np.arange(0, 16, 1)
    signal = np.zeros((5, 16))
    signal[0, :] = 1
    for layer_idx in range(1, 5):
        for chunk_idx in range(2 ** (layer_idx - 1)):
            left_logit = np.random.rand()
            left_prob = 1 / (1 + np.exp(-left_logit))
            right_prob = 1 - left_prob
            left_start = 2 * chunk_idx * 16 // 2 ** layer_idx
            left_end = (2 * chunk_idx + 1) * 16 // 2 ** layer_idx
            right_start = (2 * chunk_idx + 1) * 16 // 2 ** layer_idx
            right_end = (2 * chunk_idx + 2) * 16 // 2 ** layer_idx
            signal[layer_idx][left_start:left_end] = signal[layer_idx - 1][left_start:left_end] * left_prob
            signal[layer_idx][right_start:right_end] = signal[layer_idx - 1][right_start:right_end] * right_prob

    #  绘图
    # plt.plot(x, y_2, color='blue', linewidth=2)
    # plt.plot(x, y_3, color='yellow')
    # plt.plot(x, y_4, color='red', linewidth=2)
    # plt.plot(x, signal[1], color='blue', linewidth=5)
    # plt.plot(x, signal[2], color='blue', linewidth=5)
    plt.plot(x, (signal[1] + signal[2]) / 2.0, color='blue', linewidth=5)
    #  设置坐标系
    plt.xlim(-1.0, 17.0)
    plt.ylim(0, 1)

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    plt.axis('off')
    # ax.axes.get_yaxis().set_visible(False)
    plt.show()