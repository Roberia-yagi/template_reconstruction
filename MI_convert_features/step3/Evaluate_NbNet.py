import sys

sys.path.append('../')
sys.path.append('../../')
import os
import time
import PIL
import random
import pickle
from tqdm import tqdm
import glob
import datetime
import argparse
from typing import Any, Tuple
import matplotlib.pyplot as plt

import torch
import numpy as np
from torch import nn
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image

from sklearn.metrics import roc_curve

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

def show_graph(result_dir, cossims):
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
    plt.savefig(resolve_path(result_dir, f'{options.embedding_size}_hist.png'), bbox_inches='tight')
    plt.clf()

    # Calculate fpr, tpr, threshold
    x = np.r_[same_cossim, diff_cossim]
    y = np.r_[np.ones(same_cossim.shape), np.zeros(diff_cossim.shape)]
    fpr, tpr, thresholds = roc_curve(y, x)

    threshold_idx = np.argmin(fpr - tpr)
    threshold = thresholds[threshold_idx]

    return threshold


def normalize_hist(x, p):
    max = 0
    for item in p:
        height = item.get_height()/sum(x)
        item.set_height(height)
        if height > max:
            max = height
    return max

def calculate_inception_score(data_dir):
    transform=transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    return inception_score(IgnoreLabelDataset(dataset), device=device, batch_size=32, resize=True, splits=10)

def extract_feature(image_dir, model, transform):
    image = PIL.Image.open(image_dir)
    feature = model(transform(image).to(device).unsqueeze(0)).unsqueeze(0)
    return feature

def get_options() -> Any:
    parser = argparse.ArgumentParser()

    # Timestamp
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    parser.add_argument("--identifier", type=str, default=time_stamp, help="timestamp")
    parser.add_argument("--result_dir", type=str, default="../../../results/evaluation_results", help="path to directory which includes results")
    parser.add_argument("--dataset", type=str, required=True, help="[IJB-C, LFW, CAISA]")
    parser.add_argument("--step2_dir", type=str, required=True, help="path to directory which includes the step2 result (required to decide identities")
    parser.add_argument("--NbNet_dir", type=str, required=True, help="path to directory which includes the NbNet result")
    parser.add_argument("--embedding_size", type=int, default=512, help="dimensionality of the latent space")
    parser.add_argument("--target_model", type=str, default="FaceNet", help="target model: 'FaceNet', 'Arcface', 'Magface")
    parser.add_argument("--target_model_path", type=str, help='path to pretrained model')
    parser.add_argument("--same_pickle_path", type=str, required=True,
                        help="path to directory which includes the pickle of original data")
    parser.add_argument("--diff_pickle_path", type=str, required=True,
                        help="path to directory which includes the pickle of original data")

    opt = parser.parse_args()
    return opt

def set_global():
    global options
    global device
    options = get_options()

    gpu_idx = get_freer_gpu()
    device = f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu"

    options.device = device

def main():
    set_global()
    # Create directory to save results

    result_dir = resolve_path(options.result_dir, options.identifier)
    os.makedirs(result_dir, exist_ok=True)

    # Save options by json format
    save_json(resolve_path(result_dir, "step3.json"), vars(options))

    # Create logger
    logger = create_logger(f"Step 3", resolve_path(result_dir, "reconstruction.log"))


    # Log options
    logger.info(vars(options))

    # Load models

    img_size_T = get_img_size(options.target_model)
    T, _ = load_model_as_feature_extractor(
        arch=options.target_model,
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

    if options.dataset == 'LFW':
        dataset_dir = '../../../dataset/LFWA/lfw-deepfunneled-MTCNN160'
    elif options.dataset == 'IJB-C':
        dataset_dir = '../../../dataset/IJB-C_cropped/screened/img'
    elif options.dataset == 'CASIA':
        dataset_dir = '../../../dataset/CASIAWebFace_MTCNN160'

    # calculate Inception score
    # inception_score = calculate_inception_score(options.NbNet_dir)
    # logger.info(f'inception score is {inception_score}')

    # Compare reconstructed image with original image
    cossims_TypeA = np.array([])
    cossims_TypeB = np.array([])
    max__ = 0
    for identity_path in tqdm(glob.glob(resolve_path(options.step2_dir, '*'))):
        identity_name = remove_path_prefix(identity_path)
        if identity_name == 'best_images' or identity_name == 'original_files':
            continue
        for file_path in glob.glob(resolve_path(identity_path, '*')):
            file_name = remove_path_prefix(file_path)
            target_file_path = resolve_path(dataset_dir, identity_name, file_name)
            if not options.dataset == 'CASIA':
                target_file_path = target_file_path + '.jpg'
            target_feature = extract_feature(target_file_path, T, transform_T)

            #Type A
            NbNet_file_path = resolve_path(options.NbNet_dir, identity_name, file_name)
            if not options.dataset == 'CASIA':
                NbNet_file_path = NbNet_file_path + '.jpg'
            NbNet_feature = extract_feature(NbNet_file_path, T, transform_T)
            cossims_TypeA = np.append(cossims_TypeA, criterion(target_feature.cpu(), NbNet_feature.cpu()))

            if options.dataset == 'CASIA':
                # TypeB
                NbNet_file_paths = glob.glob(resolve_path(options.NbNet_dir, identity_name, '*'))
                NbNet_file_path_B = NbNet_file_path
                while NbNet_file_path == NbNet_file_path_B:
                    NbNet_file_path_B = random.choice(NbNet_file_paths)
                    file_path_name= remove_path_prefix(NbNet_file_path_B)
                    if file_path_name == 'best_image':
                        NbNet_file_path_B = NbNet_file_path
                        continue
                reconstructed_feature_B = extract_feature(NbNet_file_path_B, T, transform_T)
                cossims_TypeB = np.append(cossims_TypeB, criterion(target_feature.cpu(), reconstructed_feature_B.cpu()))



    threshold = show_graph(result_dir=result_dir, cossims=cossims_TypeA)
    logger.info(f'TypeA')
    logger.info(f'The average of cosine similarity: {cossims_TypeA.mean()}')
    logger.info(f'TAR: {(cossims_TypeA > threshold).sum()/len(cossims_TypeA)}')


    if options.dataset == 'CASIA':
        threshold = show_graph(result_dir=result_dir, cossims=cossims_TypeB)

        logger.info(f'TypeB')
        logger.info(f'The average of cosine similarity: {cossims_TypeB.mean()}')
        logger.info(f'TAR: {(cossims_TypeB > threshold).sum()/len(cossims_TypeB)}')

if __name__ == '__main__':
	main()
