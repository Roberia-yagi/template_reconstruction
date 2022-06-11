import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import resolve_path
identity_to_count = {}
six_digits = '([0-9][0-9][0-9][0-9][0-9][0-9]|[0-9][0-9][0-9][0-9][0-9]|[0-9][0-9][0-9][0-9]|[0-9][0-9][0-9]|[0-9][0-9]|[0-9])'
with open('/home/akasaka/nas/dataset/CelebA_MTCNN160/identity_CelebA.txt', 'r') as f:
    for s in f:
        if s.find('jpg') > 0:
            id = int(re.search(fr'.jpg {six_digits}', s).group()[4:])
            identity_to_count[id] = 0
with open('/home/akasaka/nas/dataset/CelebA_MTCNN160/identity_CelebA.txt', 'r') as f:
    for s in f:
        if s.find('jpg') > 0:
            id = int(re.search(fr'.jpg {six_digits}', s).group()[4:])
            identity_to_count[id] += 1
tmp = []
for key in identity_to_count:
    tmp.append(identity_to_count[key])
dt = pd.DataFrame(tmp)
counts_dt = dt.value_counts().sort_index()
counts_dt = counts_dt.astype(int)
counts_dt.plot.bar(x=0, y=1)
plt.savefig(resolve_path('/home/akasaka/nas/results/utils', 'CelebA_histogram.png'), )
