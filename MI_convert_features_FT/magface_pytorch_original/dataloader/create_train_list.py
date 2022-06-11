import glob
import re
import os
from tqdm import tqdm


dataset_path = '/home/akasaka/nas/dataset/faces_emore/outputs'

folder_list = glob.glob(dataset_path + '/*')

for idx, folder in enumerate(folder_list):
    # id = folder.split('/')[7][2:]
    file_list = glob.glob(folder + '/*')
    for file in file_list:
        print(f'{file} 0 {idx} 0')
