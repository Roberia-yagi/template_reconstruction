import argparse
import sys

from typing import Any
import shutil
import re
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../")
sys.path.append("../../")

from utils.util import resolve_path, save_json

def get_options() -> Any:
    parser = argparse.ArgumentParser()

    # Timestamp
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    parser.add_argument("--identifier", type=str, default=time_stamp, help="timestamp")
    parser.add_argument("--gpu_idx", type=int, default=0, help="index of cuda devices")

    parser.add_argument("--step2_dir", type=str, required=True, help="path to directory which includes results")
    
    opt = parser.parse_args()

    return opt
def main():
    options = get_options()
    # Save options by json format
    result_dir = resolve_path(options.step2_dir, 'organized_folder')
    os.makedirs(result_dir, exist_ok=True)
    six_digits = '([0-9][0-9][0-9][0-9][0-9][0-9]|[0-9][0-9][0-9][0-9][0-9]|[0-9][0-9][0-9][0-9]|[0-9][0-9][0-9]|[0-9][0-9]|[0-9])'

    flag = False
    with open(resolve_path(options.step2_dir, 'inference.log')) as f:
        for s in f:
            if s.find(f'image has been') > 0:
                label = re.search(rf'{six_digits} image', s).group()[:-5]
                label_path = resolve_path(result_dir, label)
                os.makedirs(label_path, exist_ok=True)
                flag = True
            elif flag:
                flag = False
                filename = re.search(rf'/{six_digits}]', s).group()[1:-1]
                shutil.copytree(resolve_path(options.step2_dir, filename), resolve_path(label_path, filename))

if __name__ == '__main__':
    main()