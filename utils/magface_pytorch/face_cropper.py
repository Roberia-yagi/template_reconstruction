import sys
import os

sys.path.append("../")

from glob import glob

from tqdm import tqdm 
import cv2

from util import resolve_path, get_freer_gpu
import argparse
from typing import Any
from align_faces import warp_and_crop_face, get_reference_facial_points

from retinaface.pre_trained_models import get_model

import numpy as np

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_options() -> Any:
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument("--dataset_in_dir", type=str, required=True, help="path to directory including input files")
    parser.add_argument("--dataset_out_dir", type=str, required=True, help="path to directory including output files")
    parser.add_argument("--output_size", type=int, required=True, help="path to directory including output files")
    parser.add_argument('--det-prefix', type=str, default='./model/R50', help='')
    # parser.add_argument("--original_image_size", type=int, required=True, help="image size of ")
    opt = parser.parse_args()

    return opt

def get_landmarks(img, detector):
    results = detector.predict_jsons(img)
    min_box_center = 1e9
    res = None

    if not results[0]['landmarks']:
        return None

    for result in results:
        box = result['bbox']
        landmarks = result['landmarks']
        width, height = img.shape[0], img.shape[1]
        # Center face box
        box_size_x = np.abs((box[2] - box[0])/2 + box[0] - width/2)
        box_size_y = np.abs((box[3] - box[1])/2 + box[1] - height/2)
        # print(f'box is {box}')
        # print(f'x distance from center is {box_size_x}')
        # print(f'y distance from center is {box_size_y}')
        box_size = box_size_x + box_size_y
        
        if box_size > 50:
            continue 

        if box_size < min_box_center:
            min_box_center = box_size
            res = landmarks

    return res

def main():
    options = get_options()
    device = get_freer_gpu()
    dataset_in_dir = resolve_path(options.dataset_in_dir)
    dataset_out_dir = resolve_path(options.dataset_out_dir)
    os.makedirs(dataset_out_dir, exist_ok=True)

    detector = get_model("resnet50_2020-07-20", max_size=2048, device=device)
    detector.eval()


    # Create MTCNN
    foldernames = glob(dataset_in_dir + '/*')

    # change the range to variable
    for foldername in tqdm(foldernames):
        out_folder_path = resolve_path(dataset_out_dir, foldername[foldername.rfind('/', 2)+1:])
        os.makedirs(out_folder_path, exist_ok=True)
        for filename in glob(foldername + '/*'):
            out_path = resolve_path(out_folder_path, filename[filename.rfind('/')+1:])

            # Tensor
            # img = Image.open(filename)
            # transform = transforms.ToTensor()
            # img = transform(img).unsqueeze(0)

            # Numpy
            img_raw = cv2.imread(filename, cv2.IMREAD_COLOR)
            img = np.float32(img_raw)
            landmarks = get_landmarks(img, detector)

            if landmarks is None:
                continue

            # settings
            default_square = True
            inner_padding_factor = 0.25
            outer_padding = (0, 0)

            # get the reference 5 landmarks position in the crop settings
            reference_5pts = get_reference_facial_points(
                [options.output_size, options.output_size], inner_padding_factor, outer_padding, default_square)

            # dst_img = warp_and_crop_face(raw, facial5points, reference_5pts, crop_size)
            dst_img = warp_and_crop_face(img, landmarks, reference_pts=reference_5pts, crop_size=[options.output_size, options.output_size])
            cv2.imwrite(out_path, dst_img)


if __name__ == '__main__':
	main()