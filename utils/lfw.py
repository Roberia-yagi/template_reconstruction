import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))
import torch
import glob
import PIL
from typing import Any, Callable, Optional, Tuple

from util import resolve_path, remove_path_prefix

class LFW(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dir: str,
        transform: Optional[Callable] = None,
        opencv: bool = False
    ) -> None:
        self.base_dir = base_dir
        self.folder_paths = sorted(glob.glob(resolve_path(base_dir, '*')))
        self.file_names = []
        self.labels = []
        self.transform = transform
        self.opencv = opencv

        for i, folder_path in enumerate(self.folder_paths):
            for file_path in sorted(glob.glob(resolve_path(folder_path, '*'))):
                self.labels.append(remove_path_prefix(folder_path))
                self.file_names.append(remove_path_prefix(file_path))
            
    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        data = PIL.Image.open(resolve_path(self.base_dir,
            self.file_names[index][:self.file_names[index].rfind('_')], self.file_names[index]))
        label = self.labels[index]
        filename = self.file_names[index]

        if self.transform is not None:
            data = self.transform(data)
            if self.opencv:
                data[0], data[2] = data.clone()[2], data.clone()[0]

        return data, (label, filename)

def main():
    dataset = LFW(base_dir='../../dataset/LFWA/lfw-deepfunneled-MTCNN160')
    for _, j in dataset:
        print(j)

if __name__=='__main__':
    main()
