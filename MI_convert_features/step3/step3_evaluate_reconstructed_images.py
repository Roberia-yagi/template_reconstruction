import sys

sys.path.append('../')
sys.path.append('../../')
import os
import PIL
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
from utils.util import (resolve_path, save_json, create_logger, get_img_size, load_json, 
                        load_model_as_feature_extractor, get_img_size, get_freer_gpu, remove_path_prefix)
from utils.inception_score_pytorch.inception_score import inception_score

class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index][0]

    def __len__(self):
        return len(self.orig)


def normalize_hist(x, p):
    max = 0
    for item in p:
        height = item.get_height()/sum(x)
        item.set_height(height)
        if height > max:
            max = height
    return max

def show_graph(result_dir, step1_options, cossims):
    max__ = 0
    x, _, p = plt.hist(cossims,  bins=50, alpha=0.3, color='g', edgecolor='k', label='reconstructed')
    max_ = normalize_hist(x, p)
    if max_ > max__:
        max__ = max_

    # load original cossmis from pickle and show graph of comparison between reconstructed images and original images
    with open(options.same_pickle_path, 'rb') as f:
        same_cossim = pickle.load(f)
    with open(options.diff_pickle_path, 'rb') as f:
        diff_cossim = pickle.load(f)
    x, _, p =plt.hist(same_cossim,  bins=50, alpha=0.3, color='r', edgecolor='k', label='same')
    max_ = normalize_hist(x, p)
    if max_ > max__:
        max__ = max_
    x, _, p = plt.hist(diff_cossim,  bins=50, alpha=0.3, color='b', edgecolor='k', label='diff')
    max_ = normalize_hist(x, p)
    if max_ > max__:
        max__ = max_

    plt.xlabel("Cos sim")
    plt.ylabel("Freq")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))
    plt.ylim(0, max__)
    plt.savefig(resolve_path(result_dir, f'{step1_options["target_model"]}_{options.embedding_size}_hist.png'), bbox_inches='tight')
    plt.clf()

    # Calculate fpr, tpr, threshold
    x = np.r_[same_cossim, diff_cossim]
    y = np.r_[np.ones(same_cossim.shape), np.zeros(diff_cossim.shape)]
    fpr, tpr, thresholds = roc_curve(y, x)
    print(f'FPR:{fpr}')
    print(f'TPR:{tpr}')

    # Threshold is at the crossing point
    # threshold_idx = np.argmin(fpr - tpr)
    # THreshold is at the EER
    threshold_idx = np.argmin(np.abs(1 - fpr - tpr))
    print(f'Threshold idx: {threshold_idx}')
    print(f'FPR at the threshold:{fpr[threshold_idx]}')
    print(f'TPR at the threshold:{tpr[threshold_idx]}')
    threshold = thresholds[threshold_idx]

    return threshold


def calculate_inception_score(data_dir):
    transform=transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    return inception_score(IgnoreLabelDataset(dataset), device=device, batch_size=32, resize=True, splits=10)

def get_options() -> Any:
    parser = argparse.ArgumentParser()

    # Timestamp
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    parser.add_argument("--identifier", type=str, default=time_stamp, help="timestamp")
    parser.add_argument("--result_dir", type=str, default="../../../results/evaluation_results", help="path to directory which includes results")
    parser.add_argument("--dataset", type=str, required=True, help="path to dataset directory[LFW, IJB-C, CASIA]")
    parser.add_argument("--step2_dir", type=str, required=True, help="path to directory which includes the step2 result")
    parser.add_argument("--embedding_size", type=int, default=512, help="dimensionality of the latent space")
    parser.add_argument("--target_model_path", type=str, help='path to pretrained model')
    # TODO: Pack same and diff in an argment 
    parser.add_argument("--same_pickle_path", type=str, required=True,
                        help="path to directory which includes the pickle of original data")
    parser.add_argument("--diff_pickle_path", type=str, required=True,
                        help="path to directory which includes the pickle of original data")
    parser.add_argument("--seed", type=int, default=0, help="seed for pytorch dataloader shuffle")
    parser.add_argument("--num_of_images", type=int, default=300, help="size of test dataset")

    opt = parser.parse_args()
    return opt

def set_global():
    global options
    global device
    options = get_options()

    gpu_idx = get_freer_gpu()
    device = f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu"
    print(device)

    options.device = device

def main():
    set_global()
    # Create directory to save results

    step2_dir = options.step2_dir[options.step2_dir.rfind('/'):][18:]
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

    if options.dataset == 'LFW':
        dataset_dir = '../../../dataset/LFWA/lfw-deepfunneled-MTCNN160'
    elif options.dataset == 'IJB-C':
        dataset_dir = '../../../dataset/IJB-C_cropped/screened/img'
    elif options.dataset == 'CASIA':
        dataset_dir = '../../../dataset/CASIAWebFace_MTCNN160'

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
    '''
    # IJB-C dataset
    # for reconstructed_folder_path in glob.glob(resolve_path(organized_image_folder, '*')):
    #     folder_name = reconstructed_folder_path[reconstructed_folder_path.rfind('/')+1:-1]
    #     for reconstructed_file_path in glob.glob(resolve_path(reconstructed_folder_path, '*')):
    #         file_name = reconstructed_file_path[reconstructed_file_path.rfind('/')+1:]
    #         reconstrected_best_image_path = glob.glob(resolve_path(reconstructed_file_path, 'best_image', '*'))[0]
    #         reconstructed_image = PIL.Image.open(reconstrected_best_image_path)
    #         reconstructed_feature = T(transform_T(reconstructed_image).to(device).unsqueeze(0)).unsqueeze(0)

    #         target_image_path = resolve_path(options.dataset_dir, folder_name, file_name + '.jpg')
    #         target_image = PIL.Image.open(target_image_path)
    #         target_feature = T(transform_T(target_image).to(device).unsqueeze(0)).unsqueeze(0)
    #         cossims = np.append(cossims, criterion(target_feature.cpu(), reconstructed_feature.cpu()))

    # CASIA-WebFace
    # TypeA
    for identity_path in glob.glob(resolve_path(step2_dir, '*')):
        folder_name = remove_path_prefix(identity_path)
        # best_images folder is only for inception score
        if folder_name == 'best_images':
            continue
        for filename_path in glob.glob(resolve_path(identity_path, '*')):
            file_name = remove_path_prefix(filename_path)
            reconstrected_best_image_path = glob.glob(resolve_path(filename_path, 'best_image', '*'))[0]
            reconstructed_image = PIL.Image.open(reconstrected_best_image_path)
            reconstructed_feature = T(transform_T(reconstructed_image).to(device).unsqueeze(0)).unsqueeze(0)

            target_image_path = resolve_path(dataset_dir, folder_name, file_name)
            target_image = PIL.Image.open(target_image_path)
            target_feature = T(transform_T(target_image).to(device).unsqueeze(0)).unsqueeze(0)
            cossims_TypeA = np.append(cossims_TypeA, criterion(target_feature.cpu(), reconstructed_feature.cpu()))
            break # 1 image for 1 identity for TypeA evaluation

    # TypeB
    # for identity_path in glob.glob(resolve_path(step2_dir, '*')):
    #     if not os.path.isdir(identity_path):
    #         continue
    #     folder_name = remove_path_prefix(identity_path)
    #     # best_images folder is only for inception score
    #     if folder_name == 'best_images':
    #         continue
    #     filename_path_list = glob.glob(resolve_path(identity_path, '*'))

    #     reconstructed_file_path = filename_path_list[0]
    #     reconstrected_best_image_path = glob.glob(resolve_path(reconstructed_file_path, 'best_image', '*'))[0]
    #     reconstructed_image = PIL.Image.open(reconstrected_best_image_path)
    #     reconstructed_feature = T(transform_T(reconstructed_image).to(device).unsqueeze(0)).unsqueeze(0)

    #     target_file_path = filename_path_list[1]
    #     target_filename = remove_path_prefix(target_file_path)
    #     target_image_path = resolve_path(dataset_dir, folder_name, target_filename)
    #     target_image = PIL.Image.open(target_image_path)
    #     target_feature = T(transform_T(target_image).to(device).unsqueeze(0)).unsqueeze(0)
    #     cossims_TypeB = np.append(cossims_TypeB, criterion(target_feature.cpu(), reconstructed_feature.cpu()))

    threshold = show_graph(result_dir=result_dir, step1_options=step1_options, cossims=cossims_TypeA)
    logger.info(f'TypeA')
    logger.info(f'Threshold: {threshold}')
    logger.info(f'The average of cosine similarity: {cossims_TypeA.mean()}')
    logger.info(f'TAR: {(cossims_TypeA > threshold).sum()/len(cossims_TypeA)}')

    # threshold = show_graph(result_dir=result_dir, step1_options=step1_options, cossims=cossims_TypeB)
    # logger.info(f'TypeB')
    # logger.info(f'The average of cosine similarity: {cossims_TypeB.mean()}')
    # logger.info(f'TAR: {(cossims_TypeB > threshold).sum()/len(cossims_TypeB)}')

if __name__ == '__main__':
	main()
