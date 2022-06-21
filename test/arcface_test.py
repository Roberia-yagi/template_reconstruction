import os
import sys

sys.path.append("../")
sys.path.append("../../")

import argparse
import warnings 
warnings.filterwarnings("ignore")
import numpy as np
from tqdm import tqdm
import datetime
import pickle

from typing import Any

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
from utils.celeba import CelebA
from utils.lfw import LFW
from torch import nn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from utils.util import (save_json,  create_logger, resolve_path, load_model_as_feature_extractor, 
load_model_as_feature_extractor, get_freer_gpu, get_img_size, get_output_shape)

device = f"cuda:1" if torch.cuda.is_available() else "cpu"

model, params_to_update = load_model_as_feature_extractor(
    arch='Arcface',
    embedding_size=512,
    mode='eval',
    path='',
    pretrained=True
)
model.to(device)

# Load datasets
dataset = CelebA(
    base_dir='~/nas/dataset/CelebA_MTCNN160',
    usage='all',
    transform=transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]),
    sorted=True
)

random_dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    # Optimization:
    num_workers=os.cpu_count(),
    pin_memory=True
)

criterion = nn.CosineSimilarity(dim=1)

prior_id = None
prior_feature = None
cossims = torch.tensor([])
for i, (data, id) in enumerate(dataset):
    if i == 20000:
        break
    feature = model(data.unsqueeze(0).to(device))
    if prior_id == id and prior_feature is not None:
        cossim = criterion(feature, prior_feature)
        cossims = torch.concat((cossims, cossim.detach().cpu()))
    prior_feature = feature
    prior_id = id

print(cossims)
x, _, p = plt.hist(cossims.numpy(),  color='green', bins=28, alpha=0.3, label='Same')
plt.xlabel("Cos sim")
plt.ylabel("Freq")
plt.legend()
plt.savefig(resolve_path('/home/akasaka/nas/results/utils', f'arcface_hist_1.png'))
plt.clf()