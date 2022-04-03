import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as tick

result = {
    "anet": {
        "mean": {
            "max": [45.29, 45.36, 45.08, 45.36, 45.57, 45.98, 45.76, 45.70, 45.55],
            "min": [44.92, 45.11, 44.61, 44.95, 45.35, 45.54, 45.55, 45.33, 45.27],
            "avg": [45.09, 45.24, 44.90, 45.17, 45.47, 45.70, 45.66, 45.48, 45.37]
        },
        "0.3": {
            "max": [62.27, 62.24, 62.13, 62.32, 62.94, 63.19, 62.87, 62.80, 62.41],
            "min": [61.57, 61.81, 61.36, 61.67, 62.05, 62.52, 62.69, 62.08, 61.98],
            "avg": [62.00, 62.08, 61.83, 62.02, 62.58, 62.81, 62.78, 62.43, 62.14]
        },
        "0.5": {
            "max": [45.02, 45.12, 44.78, 45.06, 45.06, 45.35, 45.42, 45.30, 44.75],
            "min": [44.08, 44.83, 44.21, 44.60, 44.94, 44.98, 44.82, 44.18, 43.98],
            "avg": [44.62, 44.98, 44.55, 44.79, 44.98, 45.10, 45.05, 44.61, 44.36]
        },
        "0.7": {
            "max": [27.77, 28.13, 27.68, 27.70, 27.97, 28.16, 28.28, 28.15, 27.72],
            "min": [27.35, 27.57, 27.35, 27.39, 27.75, 28.06, 27.72, 27.54, 27.39],
            "avg": [27.63, 27.80, 27.47, 27.56, 27.89, 28.10, 27.91, 27.82, 27.59]
        }
    },
    "charades": {
        "0.5": {
            "avg": [44.84, 45.38, 45.19, 45.34, 45.50, 46.08, 45.28, 45.75, 45.56]
        }
    },
    "tacos": {
        "0.5": {
            "avg": [33.66, 33.63, 34.52, 34.35, 34.77, 34.85, 35.74, 36.33, 35.49]
        }
    }
}

ablation = [
    [37.88, 38.73, 31.67],  # w/o mp
    [44.67, 43.31, 32.54],  # w/o signal loss
    [44.15, 43.92, 29.59],  # Pool
    [44.55, 45.22, 35.24],  # Conv
    [45.10, 46.08, 36.33]  # full
]


def draw_curve(name, metric):
    average = result[name][metric]["avg"]
    minimum = result[name][metric]["min"]
    maximum = result[name][metric]["max"]
    fig = plt.figure(dpi=500, figsize=(6, 2))
    plt.plot(range(1, len(average) + 1), average)
    ax = fig.add_subplot(111)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))

    plt.fill_between(range(1, len(minimum) + 1), minimum, maximum,
                     color='b', alpha=0.1)

    plt.xlabel("The balance factor λ", fontdict={"size": 16})
    y_label = "mIoU" if metric == "mean" else "R@1, IoU={}".format(metric)
    plt.ylabel(y_label, fontdict={"size": 16})

    plt.savefig('{}_{}.pdf'.format(name, metric[-1]), dpi=600, format='pdf', bbox_inches='tight')
    plt.show()


def draw_all_curve(metric):
    anet = np.array(result["anet"][metric]["avg"])
    charades = np.array(result["charades"][metric]["avg"])
    tacos = np.array(result["tacos"][metric]["avg"])
    fig = plt.figure(dpi=500, figsize=(8, 2))
    plt.plot(range(1, len(anet) + 1), anet / np.max(anet), label="ActivityNet Caption",
             color='#ff4b5c', marker='D', markersize=4)
    plt.plot(range(1, len(charades) + 1), charades / np.max(charades), label="Charades-STA",
             color='#056674', marker='D', markersize=4)
    plt.plot(range(1, len(tacos) + 1), tacos / np.max(tacos), label="TACoS",
             color='#66bfbf', marker='D', markersize=4)
    ax = fig.add_subplot(111)
    # plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))

    plt.xlabel("The balance factor λ", fontdict={"size": 14})
    y_label = "R@0.5 Ratio"
    plt.ylabel(y_label, fontdict={"size": 14})
    legend = ax.legend(loc='best')

    plt.savefig('hyper_{}.png'.format(metric[-1]), dpi=600, format='png', bbox_inches='tight')
    plt.show()


def scale_color(ptn):
    return ptn[0] / 255.0, ptn[1] / 255.0, ptn[2] / 255.0


def draw_bar():
    fig = plt.figure(dpi=500, figsize=(5, 2))
    tick_label = ['ActivityNet Caption', 'Charades-STA', 'TACoS']
    x = np.arange(len(tick_label))
    bar_width = 0.12
    plt.bar(x, ablation[0], bar_width, color=scale_color((237, 125, 49)), label='w/o. msg passing')
    plt.bar(x + bar_width, ablation[1], bar_width, color=scale_color((255, 192, 0)), label='w/o. signal loss')
    plt.bar(x + 2 * bar_width, ablation[2], bar_width, color=scale_color((112, 173, 71)), label='max-pooling')
    plt.bar(x + 3 * bar_width, ablation[3], bar_width, color=scale_color((91, 155, 213)), label='convolution')
    plt.bar(x + 4 * bar_width, ablation[4], bar_width, color=scale_color((68, 84, 106)), label='full')
    plt.legend(loc='best', fontsize="x-small")  # 显示图例，即label
    plt.ylim(25, 50)
    plt.ylabel("R@0.5", fontdict={"size": 10})
    plt.xticks(x + 4 * bar_width / 2, tick_label, fontsize=8)  # 显示x坐标轴的标签,即tick_label,调整位置，使其落在两个直方图中间位置
    plt.yticks(fontsize=8)  # 显示x坐标轴的标签,即tick_label,调整位置，使其落在两个直方图中间位置
    plt.savefig('bar.png', dpi=600, format='png', bbox_inches='tight')
    plt.show()


# draw_curve("anet", "0.3")
# draw_curve("anet", "0.5")
# draw_curve("anet", "0.7")
# draw_curve("anet", "mean")
# draw_all_curve("0.5")
draw_bar()
