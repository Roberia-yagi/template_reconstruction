import numpy as np
import os
import re
import torch

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

def get_freer_gpu():
    memory_available = np.empty(3)
    for i in range(3):
        os.system(f'nvidia-smi -i {i} --query-gpu=utilization.gpu --format=csv | grep %> tmp')
        lines = open('tmp', 'r').readlines()
        if lines == []:
            memory_available[i] = 100
        else:
            memory_available[i] = re.sub(r"\D", "", lines[1])

    freest_gpu = np.argmin(memory_available)

    if freest_gpu > 5:
        return -1

    for i in range(torch.cuda.device_count()):
        memory_available_before = get_memory_usage()
        device = f'cuda:{i}'
        tmp = torch.tensor([1]).to(device)
        memory_available_after = get_memory_usage()
        gpu_idx = np.argmax(memory_available_before - memory_available_after)
        del tmp
        torch.cuda.empty_cache()
        if gpu_idx == freest_gpu:
            return i