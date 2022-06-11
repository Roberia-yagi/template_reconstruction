import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))
import torch
import glob
import PIL
from typing import Any, Callable, Optional, Tuple

from util import resolve_path

class LFW(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dir: str,
        # usage: str, # 'train' | 'validate' | 'test' | 'all'
        # select: Optional[Tuple[int, int]] = (None, None), # A tuple of (num of identities, num per identity)
        # exclude: Optional[set] = None, # A set of identities which is excluded
        transform: Optional[Callable] = None,
        # sorted  = False
    ) -> None:
        self.base_dir = base_dir
        self.folder_paths= glob.glob(base_dir + '/*')
        self.file_paths = []
        self.transform = transform

        for i, folder_path in enumerate(self.folder_paths):
            for file_path in glob.glob(folder_path + '/*'):
                self.file_paths.append(file_path[file_path.rfind('/')+1:])
            
    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        data = PIL.Image.open(resolve_path(self.base_dir,
            self.file_paths[index][:self.file_paths[index].rfind('_')], self.file_paths[index]))
        label = self.file_paths[index]

        if self.transform is not None:
            data = self.transform(data)

        return data, label

def main():
    dataset = LFW(base_dir='../../dataset/LFWA/lfw-deepfunneled_MTCNN160')

if __name__=='__main__':
    main()
