import matplotlib.pyplot as plt

x = ["N=1", "N=2", "N=3", "N=4", "N=5","N=6"]
y = [0.9365, 0.9865, 0.9895, 0.9950,0.9880,0.9615]
rects = plt.barh(x, y, color=["red", "blue", "purple", "violet", "green", "black"])
for rect in rects:  # rects 是三根柱子的集合
    width = rect.get_width()
    print(width)
    plt.text(width, rect.get_y() + rect.get_height() / 2, str(width), size=10)
plt.xlim(0.0,1.3)
# plt.legend()
plt.show()

x = ["k=1e-5","k=1e-4", "k=1e-3", "k=1e-2", "k=1e-1", "k=1.0"]
y = [0.9895, 0.9900, 0.9950, 0.9885,0.9740,0.831]
# y = [0.9365, 0.9865, 0.9895, 0.9950,0.9880,0.9615]
rects = plt.barh(x, y, color=["red", "blue", "purple", "violet", "green", "black"])
for rect in rects:  # rects 是三根柱子的集合
    width = rect.get_width()
    print(width)
    plt.text(width, rect.get_y() + rect.get_height() / 2, str(width), size=10)
plt.xlim(0.0,1.3)
# plt.legend()
plt.show()