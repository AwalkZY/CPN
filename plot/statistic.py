import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import numpy as np

from utils.accessor import load_pickle


def get_count(filename):
    raw_data = load_pickle(filename + ".pkl")
    data = np.array(raw_data)
    print(filename, np.median(data))
    print(filename, np.mean(data))
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    data = data[data >= q1]
    data = data[data <= q3]
    print(filename, data.mean())
    plt.hist(data, bins=10, edgecolor="black")
    plt.show()


# get_count("anet")
# get_count("charades")
get_count("tacos")