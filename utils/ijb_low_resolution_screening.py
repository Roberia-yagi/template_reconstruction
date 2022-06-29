import sys
import os
from tqdm import tqdm

sys.path.append('../')
sys.path.append('../../')
from utils.ijb import IJB
from torchvision import transforms
dataset = IJB(
    base_dir='../../dataset/IJB-C_cropped/img',
    # transform=transforms.Compose([
        # transforms.ToTensor(),
    # ]),
)
for data, (id, filename) in tqdm(dataset):
    if data.size[0] >= 160 and data.size[1] >= 160:
        # print(data.size)
        save_folder = f'../../dataset/IJB-C_cropped/screened/img/{id}'
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        data.save(f'{save_folder}/{filename}')

