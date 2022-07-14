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

from util import resolve_path, remove_path_prefix

class CasiaWebFace(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dir: str,
        num_of_identities: int,
        num_per_identity: int,
        eval_num_of_identities: int,
        eval_num_per_identity: int,
        usage: str,
        transform: Optional[Callable] = None,
    ) -> None:

        if not usage in ['train', 'test', 'valid', 'eval']:
            raise('Casia Web Face usage error')

        self.base_dir = base_dir
        self.transform = transform

        # Initialize variables
        folder_list = glob(resolve_path(base_dir, '*'))
        self.filenames = []
        self.labels = []
        self.test_filenames = []
        self.test_labels = []
        self.eval_filenames = []
        self.eval_labels = []
        loaded_identities_counter = 0
        
        # Get a list of images which satisfies the restraint of image number
        # Filenames and Labels are training data
        # Test_filenames and test_labels are validating and testing data
        for folder in folder_list:
            file_list = glob(resolve_path(folder, '*'))
            label = remove_path_prefix(folder)
            if len(file_list) >= num_per_identity and num_of_identities > loaded_identities_counter:  
                self.filenames.extend(file_list[:num_per_identity])
                self.labels.extend([label] * num_per_identity)
                loaded_identities_counter += 1
            else:
                self.test_filenames.extend(file_list[:num_per_identity])
                self.test_labels.extend([label] * len(file_list))

        # Get a list of images for evaluation which doesn't overlap with training, validating, testing dataset
        loaded_identities_counter = 0
        for folder in folder_list:
            file_list = glob(resolve_path(folder, '*'))
            label = remove_path_prefix(folder)
            if label in self.labels:
                continue
            if len(file_list) >= eval_num_per_identity and eval_num_of_identities > loaded_identities_counter:  
                self.eval_filenames.extend(file_list[:eval_num_per_identity])
                self.eval_labels.extend([label] * eval_num_per_identity)
                loaded_identities_counter += 1


        divider = int(len(self.filenames)/10)

        if usage == 'test' or usage == 'valid':
            self.filenames = self.test_filenames
            self.labels = self.test_labels
        elif usage == 'eval':
            self.filenames = self.eval_filenames
            self.labels = self.eval_labels

        if usage == 'test':
            self.filenames = self.filenames[:divider]
            self.labels= self.labels[:divider]
        elif usage == 'valid':
            self.filenames = self.filenames[divider:divider*2]
            self.labels= self.labels[divider:divider*2]

        for i, filename in enumerate(self.filenames):
            self.filenames[i] = remove_path_prefix(filename)

        print('='*30)
        print(f'{int(len(self.filenames)/len(np.unique(self.labels)))} images per identities')
        print(f'{len(np.unique(self.labels))} identities are loaded')
        print(f'{len(self.filenames)} images are loaded')
        print('='*30)

        # Training dataset must satisfy the restraint of image number
        if usage == 'train':
            assert len(self.filenames)  == num_of_identities * num_per_identity
        elif usage == 'eval':
            assert len(self.filenames)  == eval_num_of_identities * eval_num_per_identity

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        file_path = resolve_path(self.base_dir, self.labels[index], self.filenames[index])
        data = PIL.Image.open(file_path)
        filename = self.filenames[index]
        label = self.labels[index]

        if self.transform is not None:
            data = self.transform(data)

        return data, (label, filename)


def main():
    dataset = CasiaWebFace(base_dir='../../dataset/CASIAWebFace_MTCNN160',
                           usage='eval',
                           num_of_identities=100,
                           num_per_identity=10,
                           eval_num_of_identities=300,
                           eval_num_per_identity=2,)
    print(dataset[0])

if __name__ == '__main__':
	main()