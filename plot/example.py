import matplotlib
import numpy as np
import math
import torch
import matplotlib.pyplot as plt
from scipy import interpolate

from utils.accessor import load_json


def max_min_norm(input):
    input_torch = torch.tensor(input)
    result = 0.8 * (input_torch - torch.min(input_torch)) / (torch.max(input_torch) - torch.min(input_torch))
    return result.numpy()


def draw_pred(plt, start, end):
    pred_start = max_min_norm(start)
    pred_end = max_min_norm(end)
    plt.plot(x, pred_start, color='red', linewidth=5, label='start distribution')
    plt.plot(x, pred_end, color='blue', linewidth=5, label='end distribution')


def draw_inside(plt, inside):
    # f_inside = interpolate.interp1d(x, inside, kind='linear')
    plt.plot(x, inside, color='green', linewidth=5, label='response signal')


def draw_boundary(plt, start, end, max_y, color):
    left_x = [start, start]
    right_x = [end, end]
    side_y = [0, max_y]
    side_x = [start, end]
    up_y = [max_y, max_y]
    down_y = [0, 0]
    plt.plot(left_x, side_y, color=color, linewidth=2)
    plt.plot(right_x, side_y, color=color, linewidth=2)
    plt.plot(side_x, up_y, color=color, linewidth=2)
    plt.plot(side_x, down_y, color=color, linewidth=2)


if __name__ == '__main__':
    #  自变量
    max_frame = 32
    matplotlib.pyplot.figure(figsize=(64, 11))
    x = np.arange(0, max_frame, 1)
    result_json = load_json("charades_failure.json")
    index = 7
    """
    ANET: Failure: 70
    Charades: Failure 7
    TACoS: Failure 2
    """
    """
    ANET_new:
    2:  Children that are surrounding the drum players are clapping.
    3:  The woman and dog move all around the yard while performing tricks with a frisbee.
    7:  The animated construction of fence followed by the mechanism of airflow through the build fence.
    """
    # TACoS:
    # 14: Girl twists lime on juicer
    # 15: The person juices the lime.
    # 16: Uses juice extractor on first half of lime.
    # 17: She cuts the line in half, picks up one half and removes the juice with the juice extractor.
    # 20: She repeats step 4 with the second half of the lime.
    # ANET:
    # "69: She is instructing a class."
    # "5: Children that are surrounding the drum players are clapping."
    # "16: The man does higher ones."
    # "49: The woman is grooming a dog."
    # # 5 16 49 68 69
    # # 11 12 25 31 34 38 56 *62* *69* 77
    draw_pred(plt, result_json[index]["pred_start"], result_json[index]["pred_end"])
    draw_inside(plt, result_json[index]["inside"])
    draw_boundary(plt, result_json[index]["true_start"], result_json[index]["true_end"] - 1,
                  1, "black")
    pred_start = torch.tensor(result_json[index]["pred_start"]).argmax(dim=-1)
    pred_end = torch.tensor(result_json[index]["pred_end"]).argmax(dim=-1)
    draw_boundary(plt, pred_start, pred_end, 0.8, "dimgray")
    print(result_json[index]["raw"])
    pred_start_second = pred_start * 1.0 / max_frame * result_json[index]["raw"][1]
    pred_end_second = (pred_end + 1) * 1.0 / max_frame * result_json[index]["raw"][1]
    print([pred_start_second, pred_end_second])
    # print(true_start, true_end)
    # print(torch.max(torch.tensor(pred_start), dim=-1))
    # print(torch.max(torch.tensor(pred_end), dim=-1))
    #  设置坐标系
    plt.xlim(0.0, 1.0 + max_frame)
    plt.ylim(-0.05, 1)

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    legend = ax.legend(loc='best', fontsize=56)

    plt.axis('off')
    plt.savefig('qualitative.pdf', dpi=600, format='pdf', bbox_inches='tight')
    # ax.axes.get_yaxis().set_visible(False)
    plt.show()
