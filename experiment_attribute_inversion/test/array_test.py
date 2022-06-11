import numpy as np
import time

start = time.time()
list = np.array([f"file/{num}.jpg" for num in range(int(1e8))], dtype=str)
end = time.time()
print(end - start)

start = time.time()
line = [line[line.find('/')+1: line.find('.')] for line in list]
end = time.time()
print(end - start)