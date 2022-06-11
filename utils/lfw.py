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
        self.filepaths = glob.glob(base_dir + '/*')
        self.transform = transform

        for i, filepath in enumerate(self.filepaths):
            self.filepaths[i] = filepath[filepath.rfind('/')+1:]

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        data = PIL.Image.open(resolve_path(self.base_dir, self.filepaths[index]))
        label = self.filepaths[index]

        if self.transform is not None:
            data = self.transform(data)

        return data, label

def main():
    LFW(base_dir='../../dataset/LFWA/lfw-deepfunneled-MTCNN160')

if __name__=='__main__':
    main()
