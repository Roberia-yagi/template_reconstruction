import torch
import os
import numpy as np
import re

def get_memory_usage():
    memory_available = np.empty(3)
    for i in range(3):
        os.system(f'nvidia-smi -i {i} -q -d Memory |grep -A4 GPU|grep Free >tmp')
        lines = open('tmp', 'r').readlines()
        if lines == []:
            memory_available[i] = 0
        else:
            for x in lines:
                memory_available[i] = re.sub(r"\D", "", x)

    return memory_available

print(f'Available device count: {torch.cuda.device_count()}')

for i in range(torch.cuda.device_count()):
    memory_available_before = get_memory_usage()
    device = f'cuda:{i}'
    tmp = torch.tensor([1]).to(device)
    memory_available_after = get_memory_usage()
    gpu_idx = np.argmax(memory_available_before - memory_available_after)
    print(f'Nvidia-smi {gpu_idx} can be accessed by CUDA:{i}')

del tmp
torch.cuda.empty_cache()