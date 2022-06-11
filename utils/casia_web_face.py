import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))
import torch
import PIL
from glob import glob

from torchvision import transforms

from typing import Any, Callable, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import resolve_path

class CasiaWebFace(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dir: str,
        num_of_identities: int,
        num_per_identity: int,
        usage: str,
        transform: Optional[Callable] = None,
    ) -> None:

        if not usage in ['train', 'test', 'valid']:
            raise('Casia Web Face usage error')

        self.transform = transform
        
        # For check dataset
        # file_counter = np.zeros(1000)
        # for folder in folder_list:
        #     file_list = glob(folder + '/*')
        #     file_counter[len(file_list)] += 1

        # self.df = pd.DataFrame(file_counter)
        # self.identity_size = identity_size
        # self.transform = transform

        folder_list = glob(base_dir + '/*')
        self.filenames = []
        self.labels = []
        self.test_filenames = []
        self.test_labels = []
        identities_counter = 0
        for folder in folder_list:
            file_list = glob(folder + '/*')
            if len(file_list) >= num_per_identity and num_of_identities > identities_counter:  
                self.filenames.extend(file_list[:num_per_identity])
                self.labels.extend([folder] * num_per_identity)
                identities_counter += 1
            else:
                self.test_filenames.extend(file_list[:num_per_identity])
                self.test_labels.extend([folder] * len(file_list))

        divider = int(len(self.filenames)/10)

        if usage == 'test' or usage == 'valid':
            self.filenames = self.test_filenames
            self.labels = self.test_labels

        if usage == 'test':
            self.filenames = self.filenames[:divider]
            self.labels= self.labels[:divider]
        elif usage == 'valid':
            self.filenames = self.filenames[divider:divider*2]
            self.labels= self.labels[divider:divider*2]

        print('='*30)
        print(f'{num_per_identity} images per identities')
        print(f'{len(np.unique(self.labels))} identities are loaded')
        print(f'{len(self.filenames)} images are loaded')
        print('='*30)

        if usage == 'train':
            assert len(self.filenames)  == num_of_identities * num_per_identity

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        data = PIL.Image.open(self.filenames[index])
        label = self.labels[index]

        if self.transform is not None:
            data = self.transform(data)

        return data, label


def main():
    dataset = CasiaWebFace(base_dir='../../dataset/CASIAWebFace',
                           usage='train',
                           identity_size=18)
    dataset = CasiaWebFace(base_dir='../../dataset/CASIAWebFace',
                           usage='test',
                           identity_size=18)
    dataset = CasiaWebFace(base_dir='../../dataset/CASIAWebFace',
                           usage='valid',
                           identity_size=18)

if __name__ == '__main__':
	main()