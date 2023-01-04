import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))
import torch
import glob
import PIL
from typing import Any, Callable, Optional, Tuple

from util import resolve_path, remove_path_prefix

class LFWDual(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dir1: str,
        base_dir2: str,
        transform: Optional[Callable] = None,
        opencv: bool = False
    ) -> None:
        self.base_dir1 = base_dir1
        self.base_dir2 = base_dir2
        self.folder_paths1= sorted(glob.glob(resolve_path(base_dir1, '*')))
        self.folder_paths2= sorted(glob.glob(resolve_path(base_dir2, '*')))
        self.file_names= []
        self.labels = []
        self.transform = transform
        self.opencv = opencv

        # remove unique id folder
        self.folder_paths1, self.folder_paths2 = self.remove_unique_folder(self.folder_paths1, self.folder_paths2)
        # self.folder_paths2, self.folder_paths1 = self.remove_unique_folder(self.folder_paths2, self.folder_paths1)

        for i, folder_path1 in enumerate(self.folder_paths1):
            # remove unique file from each folder
            file_paths1 = sorted(glob.glob(resolve_path(folder_path1, '*')))
            label = remove_path_prefix(folder_path1)
            file_paths2 = sorted(glob.glob(resolve_path(base_dir2, label, '*')))

            file_paths1, file_paths2 = self.remove_unique_file(file_paths1, file_paths2, label)
            # file_paths2, file_paths1 = self.remove_unique_file(file_paths2, file_paths1, label)

            # register label and filename
            for file_path1 in file_paths1:
                self.labels.append(remove_path_prefix(folder_path1))
                self.file_names.append(remove_path_prefix(file_path1))

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
        return len(self.file_paths)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        data1 = PIL.Image.open(resolve_path(self.base_dir1,
            self.file_names[index][:self.file_names[index].rfind('_')], self.file_names[index]))
        data2 = PIL.Image.open(resolve_path(self.base_dir2,
            self.file_names[index][:self.file_names[index].rfind('_')], self.file_names[index]))
        label = self.labels[index]
        filename = self.file_names[index]

        if self.transform is not None:
            data1 = self.transform(data1)
            data2 = self.transform(data2)
            if self.opencv:
                data1[0], data1[2] = data1.clone()[2], data1.clone()[0]
                data2[0], data2[2] = data2.clone()[2], data2.clone()[0]

        return (data1, data2), (label, filename)

def main():
    dataset = LFWDual(base_dir1='../../dataset/LFWA/lfw-deepfunneled-MTCNN160',
                      base_dir2='../../dataset/LFWA_MTCNN112_Arcface')
    for _, j in dataset:
        print(j)

if __name__=='__main__':
    main()
