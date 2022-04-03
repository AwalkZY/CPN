import numpy as np
import math
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
    x = np.arange(0, 16, 0.1)
    np.random.seed(100)
    #  因变量（不同均值或方差）
    # y_1 = gd(x, 3, 0.2)
    y_2 = gd(x, 2, 0.5)
    # y_3 = gd(x, 3, 5.0)
    # y_4 = gd(x, 10, 1.0)
    signal = np.zeros((5, 160))
    for layer_idx in range(5):
        for chunk_idx in range(2 ** layer_idx):
            rand_num = np.random.rand()
            start = chunk_idx * 160 // 2 ** layer_idx
            end = (chunk_idx + 1) * 160 // 2 ** layer_idx
            signal[layer_idx][start:end] = rand_num + 0.05
        signal[layer_idx] = 0.5 * (
                (signal[layer_idx] - np.min(signal[layer_idx])) /
                (np.max(signal[layer_idx]) - np.min(signal[layer_idx]))
        ) + 0.1
    signal[0] = 0
    signal[0, 30: 90] = 0.6
    #  绘图
    plt.plot(x, y_2, color='blue', linewidth=5)
    # plt.plot(x, y_3, color='yellow')
    # plt.plot(x, y_4, color='red', linewidth=2)
    # plt.plot(x, signal[0], color='green', linewidth=5)
    #  设置坐标系
    plt.xlim(-1.0, 17.0)
    plt.ylim(-0.2, 1)

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