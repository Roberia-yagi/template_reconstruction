import sys

sys.path.append('../')
sys.path.append('../../')
import os
import time
import PIL
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
                        load_model_as_feature_extractor, get_img_size, get_freer_gpu,
                        extract_target_features)
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
    parser.add_argument("--dataset_dir", type=str, default="../../../dataset/LFWA/lfw-deepfunneled-MTCNN160", help="path to dataset directory")
    parser.add_argument("--NbNet_dir", type=str, required=True, help="path to directory which includes the step2 result")
    parser.add_argument("--embedding_size", type=int, default=512, help="dimensionality of the latent space")
    parser.add_argument("--target_model", type=str, default="FaceNet", help="target model: 'FaceNet', 'Arcface', 'Magface")
    parser.add_argument("--target_model_path", type=str, help='path to pretrained model')
    parser.add_argument("--same_pickle_path", type=str, default='/home/akasaka/nas/results/show_model_acc_with_hist_LFW/2022_06_21_11_01/Magface_same_cossim.pkl',
                        help="path to directory which includes the pickle of original data")
    parser.add_argument("--diff_pickle_path", type=str, default='/home/akasaka/nas/results/show_model_acc_with_hist_LFW/2022_06_21_11_01/Magface_diff_cossim.pkl',
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

    # calculate Inception score
    inception_score = calculate_inception_score(resolve_path(options.NbNet_dir, 'best_images'))
    logger.info(f'inception score is {inception_score}')

    # Compare reconstructed image with original image
    cossims = np.array([])
    max__ = 0
    for folder_path in tqdm(glob.glob(options.NbNet_dir + '/*/')):
        folder_name = folder_path[folder_path[:-1].rfind('/')+1:-1]
        if folder_name == 'best_images':
            continue
        target_image = PIL.Image.open(resolve_path(options.dataset_dir, folder_name, folder_name + '_0001.jpg'))
        target_feature = T(transform_T(target_image).to(device).unsqueeze(0)).unsqueeze(0)
        _, reconstructed_features = extract_target_features(T, img_size_T,
            options.NbNet_dir, folder_name, True, device)
        cossims = np.append(cossims, criterion(target_feature.cpu(), reconstructed_features.cpu()))

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
    plt.savefig(resolve_path(result_dir, f'{options.target_model}_{options.embedding_size}_hist.png'), bbox_inches='tight')
    plt.clf()

    # Calculate fpr, tpr, threshold
    x = np.r_[same_cossim, diff_cossim]
    y = np.r_[np.ones(same_cossim.shape), np.zeros(diff_cossim.shape)]
    fpr, tpr, thresholds = roc_curve(y, x)

    threshold_idx = np.argmin(fpr - tpr)
    threshold = thresholds[threshold_idx]

    logger.info(f'The average of cosine similarity: {cossims.mean()}')
    logger.info(f'TAR: {(cossims > threshold).sum()/len(cossims)}')

if __name__ == '__main__':
	main()
