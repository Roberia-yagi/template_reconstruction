import numpy as np
import matplotlib
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

epochs = [1, 10, 100]
medians_list = [
	[0.000426297, 0.00073962, 0.0008868, 0.001060135],
    [0.008829989, 0.00508804, 0.002751647, 0.001954507],
    [0.304372787, 0.047645766, 0.011355574, 0.005823413]
]

betas = [0.25, 0.5, 0.75, 1.0]

for (epoch, medians) in zip(epochs, medians_list):
    plt.figure()
    plt.rcParams["font.size"] = 16
    plt.plot(betas, medians, marker='o')
    plt.xlabel("β")
    plt.ylabel("Attack accuracy")
    plt.tight_layout()
    plt.savefig(f"proposed_median_{epoch}epochs.png")

for i in font_manager.fontManager.ttflist:
    if ".ttc" in i.fname:
        print(i)