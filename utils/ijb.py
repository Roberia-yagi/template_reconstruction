import sys
import os
from matplotlib import transforms
import torchvision.transforms
sys.path.append(os.path.join(os.path.dirname(__file__)))
import torch
import glob
import PIL
from typing import Any, Callable, Optional, Tuple

from util import resolve_path

class IJB(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dir: str,
        transform: Optional[Callable] = None,
        opencv: bool = False
    ) -> None:
        self.base_dir = base_dir
        self.folder_paths= sorted(glob.glob(base_dir + '/*'))
        self.file_paths = []
        self.labels = []
        self.identities = []
        self.transform = transform
        self.opencv = opencv

        for i, folder_path in enumerate(self.folder_paths):
            identity = folder_path[folder_path.rfind('/')+1:]
            for file_path in sorted(glob.glob(folder_path + '/*')):
                self.file_paths.append(file_path[file_path.rfind('/')+1:])
                self.labels.append(folder_path[folder_path.rfind('/')+1:])
                self.identities.append(identity)

            
    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        data = PIL.Image.open(resolve_path(self.base_dir,
            self.identities[index], self.file_paths[index]))
        if data.mode != 'RGB':
            data = data.convert('RGB')
            
        label = self.labels[index]
        filename = self.file_paths[index]

        if self.transform is not None:
            data = self.transform(data)
            if self.opencv:
                data[0], data[2] = data.clone()[2], data.clone()[0]

        return data, (label, filename)

def main():
    dataset = IJB(base_dir='../../dataset/IJB-C/cropped/img', transform=torchvision.transforms.ToTensor())
    for _, _ in dataset:
        a = 1

if __name__=='__main__':
    main()
