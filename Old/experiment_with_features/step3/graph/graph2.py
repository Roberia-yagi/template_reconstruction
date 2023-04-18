import numpy as np
import matplotlib
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

epochs = [1, 10, 100]
learning_rates = ['2e-3', '2e-2', '2e-1']
medians_list = [
	# 1epoch
	[
		# lr=2e-3
        [0.000294313, 0.000526242, 0.000702976, 0.000916194],
		# lr=2e-2
        [0.000505643, 0.000738943, 0.000898081, 0.001095798],
		# lr=2e-1
        [0.00071066, 0.000762036, 0.000856545, 0.000916002],
	],
	# 10epochs
	[
        [0.000937898, 0.001148706, 0.001559658, 0.00203087],
        [0.008287019, 0.005548631, 0.002945452, 0.002021456],
        [0.014514366, 0.00386936, 0.00238315, 0.001223175],
	],
	# 100epochs
	[
        [0.009402529, 0.00653335, 0.004062839, 0.003968196],
        [0.268076181, 0.053126704, 0.010327072, 0.0057328],
        [0.765547812, 0.144793719, 0.032386392, 0.005495456],
	]
]

betas = [0.25, 0.5, 0.75, 1.0]

for (epoch, medians) in zip(epochs, medians_list):
    plt.figure()
    plt.rcParams["font.size"] = 16

    for (learning_rate, median) in zip(learning_rates, medians):
        plt.plot(betas, median, label=f'lr={learning_rate}', marker='o')

    plt.xlabel("β")
    plt.ylabel("Attack accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"proposed_median_with_lr_{epoch}epochs.png")

for i in font_manager.fontManager.ttflist:
    if ".ttc" in i.fname:
        p