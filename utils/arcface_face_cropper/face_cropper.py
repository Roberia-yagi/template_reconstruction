import sys
import os

sys.path.append("../")

from PIL import Image
from mtcnn import MTCNN
from glob import glob

from tqdm import tqdm 

from util import resolve_path, get_freer_gpu
import argparse
from typing import Any

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_options() -> Any:
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument("--dataset_in_dir", type=str, required=True, help="path to directory including input files")
    parser.add_argument("--dataset_out_dir", type=str, required=True, help="path to directory including output files")
    parser.add_argument("--dataset", type=str, required=True)
    # parser.add_argument("--original_image_size", type=int, required=True, help="image size of ")
    opt = parser.parse_args()

    return opt


def main():
    options = get_options()
    device = get_freer_gpu()
    dataset_in_dir = resolve_path(options.dataset_in_dir)
    dataset_out_dir = resolve_path(options.dataset_out_dir)
    os.makedirs(dataset_out_dir, exist_ok=True)

    # Create MTCNN
    mtcnn = MTCNN(device)

    foldernames = glob(resolve_path(dataset_in_dir, '*'))

    # change the range to variable
    for foldername in tqdm(foldernames):
        out_folder_path = resolve_path(dataset_out_dir, foldername[foldername.rfind('/', 2)+1:])
        os.makedirs(out_folder_path, exist_ok=True)
        for filename in glob(resolve_path(foldername, '*')):
            out_path = resolve_path(out_folder_path, filename[filename.rfind('/')+1:])

            img = Image.open(filename)
            if img.mode == 'RGB':
                img = mtcnn.align(img, options.dataset)
                if img is None:
                    continue
                img.save(out_path)


if __name__ == '__main__':
	main()