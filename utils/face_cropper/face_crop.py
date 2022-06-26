import sys
import os
sys.path.append("../")

from PIL import Image
from facenet_pytorch import MTCNN
from glob import glob

import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm 

from util import resolve_path
import argparse
from typing import Any

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_options() -> Any:
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument("--dataset_in_dir", type=str, required=True, help="path to directory including input files")
    parser.add_argument("--dataset_out_dir", type=str, required=True, help="path to directory including output files")
    # parser.add_argument("--original_image_size", type=int, required=True, help="image size of ")
    opt = parser.parse_args()

    return opt


def main():
    options = get_options()
    dataset_in_dir = resolve_path(options.dataset_in_dir)
    dataset_out_dir = resolve_path(options.dataset_out_dir)
    os.makedirs(dataset_out_dir, exist_ok=True)

    # image_crop_size = 100

    # Create MTCNN
    mtcnn = MTCNN()

    foldernames = glob(dataset_in_dir + '/img')

    # change the range to variable
    for foldername in tqdm(foldernames):
        out_folder_path = resolve_path(dataset_out_dir, foldername[foldername.rfind('/', 2)+1:])
        os.makedirs(out_folder_path, exist_ok=True)
        for filename in glob(foldername + '/*'):
            print(filename)
            out_path = resolve_path(out_folder_path, filename[filename.rfind('/')+1:])

            img = Image.open(filename)
            if img.mode == 'RGB':
                mtcnn(img, save_path=out_path)

if __name__ == '__main__':
	main()