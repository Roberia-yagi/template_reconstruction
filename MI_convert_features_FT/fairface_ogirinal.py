import os
import csv
import random
import pandas
import torch
import torchvision
import PIL
import numpy as np
from torch.utils.data.dataset import Subset
from sklearn.model_selection import train_test_split

from torchvision import transforms

from collections import Counter
from typing import Any, Callable, Optional, Tuple, Dict

from util import resolve_path

class Fairface(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dir: str,
        usage: str, # 'train' | 'val'
        data_num: int,
        attributes: set,
        transform: Optional[Callable] = None
    ) -> None:
        self.base_dir = base_dir
        self.usage = usage
        self.transform = transform
        self.data_num=data_num 
        self.attributes=attributes

        data = pandas.read_csv(
            resolve_path(self.base_dir, f'fairface_label_{usage}.csv'),
            delim_whitespace=False,
            header=0,
            index_col=0
        )

        # Slice dataset by attributes
        if attributes is not None:
            for attribute in attributes:
                for column in data:
                    param = (data[column] == attribute)
                    if any(param):
                        data = data[param].copy()

        # Slice dataset to specific size
        if data_num is not None:
            data = data[0:self.data_num]

        # Store dataset on memory
        self.filenames = data.index.values
        self.labels = [line[line.find('/')+1: line.find('.')] for line in data.index.values]

        # # Show settings
        # print("="*64)
        # print("Fairface dataset")
        # print(f"usage:\t{usage}")
        # print(f"num of files:\t{len(self.filenames)}")
        # print("="*64)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        data = PIL.Image.open(resolve_path(self.base_dir, self.filenames[index]))
        label = self.labels[index]

        if self.transform is not None:
            data = self.transform(data)

        return data, label


def main():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = Fairface(
        base_dir='/home/akasaka/nas/dataset/fairface',
        usage='train',
        transform=transform,
        data_num=1000,
        attributes={'50-59', 'Male'}
    )

if __name__ == '__main__':
	main()