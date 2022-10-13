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
    parser.add_argument("--dataset", type=str, required=True, help="[IJB-C, LFW, CAISA]")
    parser.add_argument("--step2_dir", type=str, required=True, help="path to directory which includes the step2 result (required to decide identities")

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

    if options.dataset == 'LFW':
        dataset_dir = '../../../dataset/LFWA/lfw-deepfunneled-MTCNN160'
    elif options.dataset == 'IJB-C':
        dataset_dir = '../../../dataset/IJB-C_cropped/screened/img'
    elif options.dataset == 'CASIA':
        dataset_dir = '../../../dataset/CASIAWebFace_MTCNN160'

    for identity_path in glob.glob(resolve_path(options.step2_dir, '*')):
        identity_name = remove_path_prefix(identity_path)
        if str.isdigit(identity_name):
            print(identity_name)

if __name__ == '__main__':
	main()
