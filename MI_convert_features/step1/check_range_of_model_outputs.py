import sys

sys.path.append("../")
sys.path.append("../../")

import os
import argparse
from termcolor import cprint
from tqdm import tqdm
import datetime
from typing import Any, List, Tuple


import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.fairface import Fairface

from utils.util import (save_json, load_json, create_logger, resolve_path, load_model_as_feature_extractor, BatchLoader,
    get_freer_gpu, get_img_size, extract_features_from_nnModule)

def get_options() -> Any:
    parser = argparse.ArgumentParser()

    # Timestamp
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    parser.add_argument("--identifier", type=str, default=time_stamp, help="timestamp")
    parser.add_argument("--gpu_idx", type=int, default=0, help="index of cuda devices")

    # Save
    parser.add_argument("--save", type=bool, default=True, help="if save log and data")

    # Directories
    parser.add_argument("--result_dir", type=str, default="~/nas/results/MI_convert_features/step1", help="path to directory which includes results")
    parser.add_argument("--dataset_dir", type=str, default="~/nas/dataset/fairface", help="path to directory which includes dataset(Fairface)")
    parser.add_argument('--target_model_path', type=str, help='path to pretrained target model')

    # Conditions
    parser.add_argument("--embedding_size", type=int, default="512", help="embedding size of features of target model:[128, 512]")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--target_model", type=str, default="FaceNet", help="target model: 'FaceNet', 'Arcface', 'Magface")
    parser.add_argument("--data_num", type=int, default=1000000, help="number of data to use")

    opt = parser.parse_args()

    return opt

def load_fairface_dataset_loader_for_all(
    base_dir: str,
    usage: str,
    data_num: int,
    options: Any,
) -> Tuple[Fairface, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = Fairface(
        base_dir=base_dir,
        usage=usage,
        transform=transform,
        attributes=None,
        data_num=data_num,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=options.batch_size,
        shuffle=False,
        # Optimization:
        num_workers=os.cpu_count(),
        pin_memory=True
    )

    return dataset, dataloader


def set_global():
    global options
    global device

    options = get_options()
    # Decide device
    # gpu_id = get_freer_gpu()
    # device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    # Decide device
    device = f"cuda:{options.gpu_idx}" if torch.cuda.is_available() else "cpu"

    options.device = device

def main():

    set_global()

    # Load models
    img_size = get_img_size(options.target_model)

    model, _ = load_model_as_feature_extractor(
        arch=options.target_model,
        embedding_size=options.embedding_size,
        mode='eval',
        path=options.target_model_path,
        pretrained=True
    )

    if isinstance(model, nn.Module):
        model.to(device) 

    _, dataloader = load_fairface_dataset_loader_for_all(options.dataset_dir, \
                                                'train', \
                                                options.data_num, \
                                                options)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
    ])

    max_ = -1e9
    min_ = 1e9

    for batch, _ in tqdm(dataloader):
        batch = batch.to(device)

        features = model(transform(batch))
        if max_ < torch.max(features):
            max_ = torch.max(features) 
        if min_ > torch.min(features):
            min_ = torch.min(features) 

    cprint(f'max value in {options.target_model} is {max_}', 'green')
    cprint(f'min value in {options.target_model} is {min_}', 'green')

if __name__ == '__main__':
    main()