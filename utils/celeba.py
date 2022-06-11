import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))
import pandas
import torch
from tqdm import tqdm
import PIL
import numpy as np

from torchvision import transforms

from collections import Counter
from typing import Any, Callable, Optional, Tuple, Dict

from util import resolve_path

class CelebA(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dir: str,
        usage: str, # 'train' | 'validate' | 'test' | 'all' select: Optional[Tuple[int, int]] = (None, None), # A tuple of (num of identities, num per identity)
        select: Optional[Tuple[int, int]] = (None, None), # A tuple of (num of identities, num per identity)
        filter: list = None,
        transform: Optional[Callable] = None,
        sorted  = False
    ) -> None:

        self.base_dir = base_dir
        self.transform = transform

        if not usage in ['train', 'validate', 'test', 'all']:
            raise("Usage should be 'train', 'validate', 'test', or 'all")

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

        #TODO: refact all codes to adopt this format (concated eval_p and identities)
        if sorted:
            eval_partitions = eval_partitions.rename(columns={0:'filename', 1:'partition'})
            identities = identities.rename(columns={0:'filename', 1:'identity'})
            df_for_sorted = eval_partitions.join(identities)
            df_for_sorted = df_for_sorted.sort_values(by='identity')
            # Create dataset mask for the usage
            if usage == 'train':
                mask = df_for_sorted['partition'] == 0
            elif usage == 'validate':
                mask = df_for_sorted['partition'] == 1
            elif usage == 'test':
                mask = df_for_sorted['partition'] == 2
            elif usage == 'all':
                mask = slice(None) # Use all data

        else:
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
            elif usage == 'all':
                mask = slice(None) # Use all data

        #TODO: refactor all codes to adopt this format (concated eval_p and identities)
        if sorted:
            self.filenames = df_for_sorted[mask].index.values
            self.identities = df_for_sorted[mask]['identity'].values[:, np.newaxis]
            self.identity_set = set(identity[0] for identity in self.identities)
        else:
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

        # Select data which satisfies the 'select' constraint
        if select[0] is not None and select[1] is not None:
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

            #TODO: Delete
            print(len(identity_set))

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

        if filter is not None:
            self.filenames = self.filenames[filter]
            self.identities = self.identities[filter]
            self.labels = self.labels[filter]
            print(f"Post modified:\tidentities num ???, data num {len(self.identities)}")
        else:
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
    dataset = CelebA(
        base_dir="~/nas/dataset/CelebA_MTCNN160",
        usage='train',
        select=(10000, 5),
        transform=transforms.Compose([
            transforms.Resize((120, 120)),
            transforms.ToTensor(),
        ]),
        sorted=False
    )
    print(len(dataset))
    

if __name__ == '__main__':
	main()