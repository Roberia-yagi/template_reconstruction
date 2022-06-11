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

class CelebA(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dir: str,
        usage: str, # 'train' | 'validate' | 'test'
        select: Optional[Tuple[int, int]] = None, # A tuple of (num of identities, num per identity)
        exclude: Optional[set] = None, # A set of identities which is excluded
        transform: Optional[Callable] = None
    ) -> None:
        self.base_dir = base_dir
        self.transform = transform

        # Load csv files of CelebA
        eval_partitions = pandas.read_csv(
            resolve_path(self.base_dir, 'list_eval_partition.txt'),
            delim_whitespace=True,
            header=None,
            index_col=0
        )
        identities = pandas.read_csv(
            resolve_path(self.base_dir, 'identity_CelebA.txt'),
            delim_whitespace=True,
            header=None,
            index_col=0
        )

        # Create dictionary[identity] = filenames
        self.identity_filenames = dict()
        for index, row in enumerate(identities.iterrows()):
            filename = str(row[0]).zfill(6)
            identity = row[1].iloc[-1]
            if identity in self.identity_filenames:
                self.identity_filenames[identity].append(filename)
            else:
                self.identity_filenames[identity] = [filename]

        # Create dataset mask for the usage
        if usage == 'train':
            mask = eval_partitions[1] == 0
        elif usage == 'validate':
            mask = eval_partitions[1] == 1
        elif usage == 'test':
            mask = eval_partitions[1] == 2
        else:
            mask = slice(None) # Use all data
         
        # Store dataset on memory
        self.filenames = eval_partitions[mask].index.values
        self.identities = identities[mask].values
        self.identity_set = set(identity[0] for identity in self.identities)

        # Show settings
        print("="*64)
        print("CelebA dataset")
        print(f"usage:\t{usage}")
        print(f"num of identities:\t{'none' if select is None else select[0]}")
        print(f"num per identity:\t{'none' if select is None else select[1]}")
        print("-"*32)
        print(f"Pre modified:\tidentities num {len(self.identity_set)}, data num {len(self.identities)}")


        # Exclude data which is included 'exclude' set
        if exclude is not None:
            filenames = []
            identities = []
            for index, identity in enumerate(self.identities):
                if identity[0] not in exclude:
                    filenames.append(self.filenames[index])
                    identities.append(self.identities[index])

            self.filenames = np.array(filenames)
            self.identities = np.array(identities)
            self.identity_set = set([identity[0] for identity in self.identities])

            print(f"Excluded:\tidentities num {len(self.identity_set)}, data num {len(self.identities)}")

        # Select data which satisfies the 'select' constraint
        if select is not None:
            num_of_identities = select[0]
            num_per_identity = select[1]

            # Count num per identity
            identity_counter = Counter()
            for identity in self.identities:
                identity_counter[identity[0]] += 1

            # Select identities which satisfy the num per identity
            identity_set = set()
            for identity, count in identity_counter.most_common():
                if count >= num_per_identity and len(identity_set) < num_of_identities:
                    identity_set.add(identity)

            # Create dataset which satisfies the constraint
            filenames = []
            identities = []
            identity_counter.clear()
            for index, identity in enumerate(self.identities):
                if identity[0] in identity_set and identity_counter[identity[0]] < num_per_identity:
                    filenames.append(self.filenames[index])
                    identities.append(self.identities[index])
                    identity_counter[identity[0]] += 1

            # Update dataset
            self.filenames = np.array(filenames)
            self.identities = np.array(identities)
            self.identity_set = identity_set

            print(f"Selected:\tidentities num {len(self.identity_set)}, data num {len(self.identities)}")
        
        self.labels = np.array([identity[0] - 1 for identity in self.identities])
        
        print(f"Post modified:\tidentities num {len(self.identity_set)}, data num {len(self.identities)}")
        print("="*64)

    def __len__(self) -> int:
        return len(self.identities)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        data = PIL.Image.open(resolve_path(self.base_dir, "img_align_celeba", self.filenames[index]))
        label = self.labels[index]

        if self.transform is not None:
            data = self.transform(data)

        return data, label

    def get_images_by_identity(self, index: int) -> list:
        images = []
        for filename in self.identity_filenames[index]:
            image = PIL.Image.open(resolve_path(self.base_dir, "img_align_celeba", filename))
            convert_tensor = transforms.ToTensor()
            converted_image = convert_tensor(image)
            images.append(converted_image)

        return images


def main():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    base_dir='~/nas/dataset/CelebA_MTCNN160/'
    private_dataset = CelebA(
        base_dir=base_dir,
        usage='train',
        select=(1000, 30),
        transform=transform
    )

    test_dataset = CelebA(
        base_dir=base_dir,
        usage='test',
        transform=transform
    )

    public_dataset = CelebA(
        base_dir=base_dir,
        usage='train',
        exclude=private_dataset.identity_set,
        transform=transform
    )

    training_indices, test_indices = train_test_split(
        list(range(len(private_dataset))),
        test_size=0.1,
        stratify=private_dataset.labels,
        random_state=0
    )


    # training_dataset = Subset(private_dataset, training_indices)
    # test_dataset = Subset(private_dataset, test_indices)

    # train_counter = Counter()
    # for image, label in training_dataset:
    #     train_counter[label] += 1
    
    # print(train_counter)
    # print(len(train_counter))

    # test_counter = Counter()
    # for image, label in test_dataset:
    #     test_counter[label] += 1
    
    # print("="*64)
    
    # print(test_counter)
    # print(len(test_counter))

    # dataloader = torch.utils.data.DataLoader(
    #     private_dataset,
    #     batch_size=64,
    #     num_workers=8
    # )

    # for images, labels in dataloader:
    #     break

if __name__ == '__main__':
	main()