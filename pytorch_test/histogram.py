import numpy as np
import matplotlib.pyplot as plt
a = np.random.randn(10000)
# plt.figure()
plt.hist(a-1, bins=100, alpha=0.3, color='r', label='a')
plt.hist(a+1, bins=100, alpha=0.3, color='b', label='b')
plt.xlabel("Cos sim")
plt.ylabel("Freq")
plt.legend()
plt.savefig('./results.png')
plt.show()
plt.clf()