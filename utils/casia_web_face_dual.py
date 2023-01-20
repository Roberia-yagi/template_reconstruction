import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))
import torch
import PIL
from glob import glob

from typing import Any, Callable, Optional, Tuple
import numpy as np

from util import resolve_path, remove_path_prefix

class CasiaWebFaceDual(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dir1: str,
        base_dir2: str,
        num_of_identities: int,
        num_per_identity: int,
        usage: str,
        transform: Optional[Callable] = None,
        eval_num_of_identities = None,
        eval_num_per_identity = None,
    ) -> None:

        if not usage in ['train', 'test', 'valid', 'eval']:
            raise('Casia Web Face usage error')

        self.base_dir1= base_dir1
        self.base_dir2= base_dir2 
        self.transform = transform

        # Initialize variables
        folder_paths1 = glob(resolve_path(base_dir1, '*'))
        folder_paths2 = glob(resolve_path(base_dir2, '*'))
        self.filenames = []
        self.labels = []
        self.test_filenames = []
        self.test_labels = []
        self.eval_filenames = []
        self.eval_labels = []
        loaded_identities_counter = 0


        folder_paths1, folder_paths2 = self.remove_unique_folder(folder_paths1, folder_paths2)
        folder_paths2, folder_paths1 = self.remove_unique_folder(folder_paths2, folder_paths1)

        # Get a list of images which satisfies the restraint of image number
        # Filenames and Labels are training data
        # Test_filenames and test_labels are validating and testing data
        for folder_path1 in folder_paths1:
            # The same identity folder should be in both lists
            file_paths1 = glob(resolve_path(folder_path1, '*'))
            label = remove_path_prefix(folder_path1)
            file_paths2 = glob(resolve_path(base_dir2, label, '*'))

            file_paths1, file_paths2 = self.remove_unique_file(file_paths1, file_paths2, label)
            file_paths2, file_paths1 = self.remove_unique_file(file_paths2, file_paths1, label)

            if len(file_paths1) >= num_per_identity:
                if num_of_identities > loaded_identities_counter:
                    self.filenames.extend(file_paths1[:num_per_identity])
                    self.labels.extend([label] * num_per_identity)
                    loaded_identities_counter += 1
                else:
                    self.test_filenames.extend(file_paths1[:num_per_identity])
                    # self.test_labels.extend([label] * len(files))
                    self.test_labels.extend([label] * num_per_identity)

        # Get a list of images for evaluation which doesn't overlap with training, validating, testing dataset
        if usage == 'eval':
            loaded_identities_counter = 0
            for folder_path1 in folder_paths1:
                file_paths1 = glob(resolve_path(folder_path1, '*'))
                label = remove_path_prefix(folder_path1)
                if label in self.labels:
                    continue
                if len(file_paths1) >= eval_num_per_identity and eval_num_of_identities > loaded_identities_counter:  
                    self.eval_filenames.extend(file_paths1[:eval_num_per_identity])
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

    def remove_unique_folder(self, folder_paths1, folder_paths2):
        for folder_path1 in folder_paths1:
            label = remove_path_prefix(folder_path1)
            folder_path2 = resolve_path(self.base_dir2, label)
            if not folder_path2 in folder_paths2:
                folder_paths1.remove(folder_path1)
        return folder_paths1, folder_paths2

    def remove_unique_file(self, file_paths1, file_paths2, label):
        file_paths = file_paths1.copy()
        for i, file_path1 in enumerate(file_paths):
            id = remove_path_prefix(file_path1)
            file_path2 = resolve_path(self.base_dir2, label, id)
            if not file_path2 in file_paths2:
                file_paths1.remove(file_path1)
        return file_paths1, file_paths2

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        file_path1 = resolve_path(self.base_dir1, self.labels[index], self.filenames[index])
        file_path2 = resolve_path(self.base_dir2, self.labels[index], self.filenames[index])
        data1 = PIL.Image.open(file_path1)
        data2 = PIL.Image.open(file_path2)
        filename = self.filenames[index]
        label = self.labels[index]

        if self.transform is not None:
            data1 = self.transform(data1)
            data2 = self.transform(data2)

        return (data1, data2), (label, filename)


def main():
    dataset = CasiaWebFaceDual(base_dir1='../../dataset/CASIAWebFace_MTCNN160_Facenet',
                           base_dir2='../../dataset/CASIAWebFace_MTCNN112_Arcface_without_screening',
                           usage='train',
                           num_of_identities=5200,
                           num_per_identity=20,
                           eval_num_of_identities=300,
                           eval_num_per_identity=2,)

    for _ in dataset:
        break



if __name__ == '__main__':
	main()