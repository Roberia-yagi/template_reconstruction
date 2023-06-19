import sys

sys.path.append('../')
sys.path.append('../../')
import os
import PIL
import json
import pickle
import glob
import datetime
import argparse
from typing import Any
import matplotlib.pyplot as plt

import torch
import numpy as np
from torch import nn
from torchvision import transforms
from torchvision import datasets
from sklearn.metrics import roc_curve

from utils.lfw import LFW
from utils.ijb import IJB
from utils.util import (resolve_path, save_json, create_logger, align_face_image, load_json, 
                        load_model_as_feature_extractor, get_img_size, set_global, remove_path_prefix)
from utils.inception_score_pytorch.inception_score import inception_score
from utils.arcface_face_cropper.mtcnn import MTCNN

class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index][0]

    def __len__(self):
        return len(self.orig)

def get_options() -> Any:
    parser = argparse.ArgumentParser()

    # Timestamp
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    parser.add_argument("--gpu_idx", type=int, default=None)
    parser.add_argument("--identifier", type=str, default=time_stamp, help="timestamp")
    parser.add_argument("--result_dir", type=str, default="../../../results/evaluation_results", help="path to directory which includes results")
    parser.add_argument("--step2_dir", type=str, required=True, help="path to directory which includes the step2 result")
    parser.add_argument("--embedding_size", type=int, default=512, help="dimensionality of the latent space")
    parser.add_argument("--target_model_path", type=str, help='path to pretrained model')
    parser.add_argument("--threshold_path", type=str, required=True,
                        help="path to directory which includes")
    parser.add_argument("--seed", type=int, default=0, help="seed for pytorch dataloader shuffle")
    parser.add_argument("--num_of_images", type=int, default=300, help="size of test dataset")

    opt = parser.parse_args()
    return opt


def get_thresholds():
    threshold_path = resolve_path(options.threshold_path, 'thresholds.json')
    with open(threshold_path) as f:
        thresholds = json.load(f)
    return thresholds

def calculate_inception_score(data_dir):
    transform=transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    return inception_score(IgnoreLabelDataset(dataset), device=device, batch_size=32, resize=True, splits=10)


def main():
    global options
    global device
    device, options = set_global(get_options)
    # Create directory to save results

    step2_dir = options.step2_dir[options.step2_dir.rfind('/'):]
    result_dir = resolve_path(options.result_dir, (options.identifier + '_' + step2_dir))
    os.makedirs(result_dir, exist_ok=True)

    step1_options = load_json(resolve_path(options.step2_dir, "step1.json"))
    step2_options = load_json(resolve_path(options.step2_dir, "step2.json"))
    
    # Save options by json format
    save_json(resolve_path(result_dir, "step1.json"), step1_options)
    save_json(resolve_path(result_dir, "step2.json"), step2_options)
    save_json(resolve_path(result_dir, "step3.json"), vars(options))

    # Create logger
    logger = create_logger(f"Step 3", resolve_path(result_dir, "reconstruction.log"))

    step2_dir = options.step2_dir
    step2_options = load_json(resolve_path(step2_dir, "step2.json"))

    # Log options
    logger.info(vars(options))

    # Load models
    img_size_T = get_img_size(step1_options['target_model'])

    T, _ = load_model_as_feature_extractor(
        arch=step1_options['target_model'],
        embedding_size=options.embedding_size,
        mode='eval',
        path=options.target_model_path,
        pretrained=True
    )
    detector = MTCNN(device)

    if isinstance(T, nn.Module):
        T.to(device) 
        T.eval()

    transform_T=transforms.Compose([
        transforms.Resize((img_size_T, img_size_T)),
        transforms.ToTensor()
    ]) 

    criterion = torch.nn.CosineSimilarity(dim=2)

    # calculate Inception score
    inception_score = calculate_inception_score(resolve_path(options.step2_dir, 'best_images'))
    logger.info(f'inception score is {inception_score}')

    # Compare reconstructed image with original image
    cossims_TypeA = np.array([])
    cossims_TypeB = np.array([])

    '''
    TODO: dataloader -> dataset
    data: tensor of image
    label: identity (100 or Akasaka)
    filename: filename of image (16523.jpg)
    folder_name = label
    organized_image_folder = resolve_path(options.step2_dir, 'organized_folder')
    root - Anil(identity) - Anli1(file)   - reconstructed_image_1.jpg
                                          - reconstructed_image_2.jpg
                                          - best_images
         - Tylor(identity) - Tylor1(file) - reconstructed_image_1.jpg
                                          - reconstructed_image_2.jpg
                                          - best_images
    '''

    # For all
    identity_paths = glob.glob(resolve_path(step2_dir, '*')) # /.../Aaron_Guiel
    for identity_path in identity_paths:
        folder_name = remove_path_prefix(identity_path) # Aaron_Guiel
        # best_images folder is only for inception score
        if folder_name == 'best_images':
            continue
        filename_paths = glob.glob(resolve_path(identity_path, '*')) # /.../Aaron_Guiel/Aaron_Guiel0001
        for filename_path in filename_paths:
            # Extract a feature from the reconstructed image
            file_name = remove_path_prefix(filename_path) # Aaron_Guiel0001
            reconstrected_best_image_path = glob.glob(resolve_path(filename_path, 'best_image', '*'))[0] # /.../Aaron_Guiel/Aaron_Guiel0001/best_images
            reconstructed_image = transform_T(PIL.Image.open(reconstrected_best_image_path))
            aligned_reconstructed_image = align_face_image(reconstructed_image, 'GAN', step1_options['target_model'], detector)
            if aligned_reconstructed_image is None:
                logger.info('the reconstructed image cannot be aligned')
            reconstructed_feature = T(aligned_reconstructed_image.to(device)).unsqueeze(0)

            # TypeA
            # Extract a feature from the counterpart of the target image
            target_image_path_typeA = resolve_path(step2_options.dataset_dir, folder_name, file_name)
            target_image = transform_T(PIL.Image.open(target_image_path_typeA))
            target_feature = T(target_image.to(device).unsqueeze(0)).unsqueeze(0)
            cossims_TypeA = np.append(cossims_TypeA, criterion(target_feature.cpu(), reconstructed_feature.cpu()))

            # TypeB
            target_image_paths = glob.glob(resolve_path(step2_options.dataset_dir, folder_name, '*'))
            for target_image_path_typeB in target_image_paths:
                if target_image_path_typeB == target_image_path_typeA:
                    continue
                target_image = transform_T(PIL.Image.open(target_image_path_typeB))
                target_feature = T(target_image.to(device).unsqueeze(0)).unsqueeze(0)
                cossims_TypeB = np.append(cossims_TypeB, criterion(target_feature.cpu(), reconstructed_feature.cpu()))
                break
            break

    thresholds = get_thresholds()
    threshold_EER = float(thresholds['EER']['threshold'])
    logger.info(f'Threshold at EER: {threshold_EER}')
    logger.info(f'TypeA')
    logger.info(f'The average of cosine similarity: {cossims_TypeA.mean()}')
    logger.info(f'TAR at EER: {(cossims_TypeA > threshold_EER).sum()/len(cossims_TypeA)}')

    logger.info(f'TypeB')
    logger.info(f'The average of cosine similarity: {cossims_TypeB.mean()}')
    logger.info(f'TAR at EER: {(cossims_TypeB > threshold_EER).sum()/len(cossims_TypeB)}')

if __name__ == '__main__':
	main()
