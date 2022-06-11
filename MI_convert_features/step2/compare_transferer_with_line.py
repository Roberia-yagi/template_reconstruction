import argparse
import sys

from typing import Any
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

    parser.add_argument("--result_dir", type=str, default="~/nas/results/loss_history", help="path to directory which includes results")
    parser.add_argument("--log_paths", nargs='+', required=True, help="path to directory which includes results")
    
    opt = parser.parse_args()

    return opt
def main():

    options = get_options()
    # Save options by json format
    result_dir = resolve_path(options.result_dir, options.identifier)
    os.makedirs(result_dir, exist_ok=True)
    save_json(resolve_path(result_dir, "step1.json"), vars(options))
    base_epoch_count = 0
    three_digits = '([0-9][0-9][0-9]|[0-9][0-9]|[0-9])'
    six_digits = '([0-9][0-9][0-9][0-9][0-9][0-9]|[0-9][0-9][0-9][0-9][0-9]|[0-9][0-9][0-9][0-9]|[0-9][0-9][0-9]|[0-9][0-9]|[0-9])'
    for log in options.log_paths:
        train_loss_history = np.array([])
        valid_loss_history = np.array([])
        test_loss_history = np.array([])
        img_count = 0
        epoch_count=0
        with open(resolve_path(log, 'training.log')) as f:
            for s in f:
                if s.find('Epoch') > 0:
                    epoch = int(re.search(fr'/\s*{three_digits}', s).group()[1:])
                    epoch_count += 1
                elif s.find('[Train]') > 0:
                    img_count = int(re.search(fr'/\s*{six_digits}', s).group()[1:])
                elif s.find('Train avg') > 0:
                    loss = float(re.search(r'.\.......', s).group())
                    train_loss_history = np.append(train_loss_history, loss)
                elif s.find('Valid avg') > 0:
                    loss = float(re.search(r'.\.......', s).group())
                    valid_loss_history = np.append(valid_loss_history, loss)
                elif s.find('Test avg') > 0:
                    loss = float(re.search(r'.\.......', s).group())
                    test_loss_history = np.append(test_loss_history, loss)

        print(img_count)
        plt.plot(list(range(base_epoch_count, epoch_count)), test_loss_history, label=img_count)

    # print(test_loss_history)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(resolve_path(result_dir, f'loss.png'), )

if __name__ == '__main__':
    main()