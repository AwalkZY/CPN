import matplotlib.pyplot as plt

frame = {
    "anet": [5, 6, 7],
    "tacos": [6, 7, 8],
    "charades": [4, 5, 6]
}
miou = {
    "anet": [44.52, 45.70, 44.74],
    "charades": [43.46, 43.90, 41.44],
    "tacos": [30.39, 34.49, 33.47]
}

fig = plt.figure(dpi=500, figsize=(6, 2))
ax = fig.add_subplot(111)
ax.set_xticks([4, 5, 6, 7, 8])

# 设置线宽
plt.plot(frame["anet"], miou["anet"], linewidth=2, label="ActivityNet Caption",
         color='#ff4b5c', marker='D', markersize=4)

plt.plot(frame["charades"], miou["charades"], linewidth=2, label="Charades-STA",
         color='#056674', marker='D', markersize=4)

plt.plot(frame["tacos"], miou["tacos"], linewidth=2, label="TACoS",
         color='#66bfbf', marker='D', markersize=4)

# 设置图表标题，并给坐标轴添加标签
plt.ylabel("mIoU", fontsize=12)
plt.xlabel("The logarithm of frame number to base 2", fontsize=12)

# 设置坐标轴刻度标记的大小
# plt.tick_params(axis='both', labelsize=10)
legend = ax.legend(loc='best')
plt.savefig('frame.png', dpi=600, format='png', bbox_inches='tight')
plt.show()
