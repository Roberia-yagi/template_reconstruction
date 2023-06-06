import sys
from unittest import result
import pandas as pd
sys.path.append("../")
sys.path.append("../../")

import os
import argparse
import pickle
from tqdm import tqdm 
import datetime
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from sklearn.decomposition import PCA 

from utils.lfw import LFW
from utils.celeba import CelebA 
from utils.util import (save_json, load_json, load_model_as_feature_extractor,
create_logger, resolve_path, get_img_size)

def get_options() -> Any:
    parser = argparse.ArgumentParser()

    # Timestamp
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    parser.add_argument("--identifier", type=str, default=time_stamp, help="name of result folder")
    parser.add_argument("--dataset_dir", type=str, default="../../../dataset/CelebA_MTCNN160", help="path to directory which includes dataset(Fairface)")
    parser.add_argument("--gpu_idx", type=int, default=0, help="index of cuda devices")
    parser.add_argument("--result_dir", type=str, default="../../../results/pca", help="path to directory which includes results")
    parser.add_argument("--target_model", type=str, default="FaceNet", help="target model: 'FaceNet', 'Arcface', 'Magface")
    parser.add_argument("--embedding_size", type=int, default=512, help="embedding size of features of target model:[128, 512]")
    parser.add_argument('--dataset', type=str, default="CelebA", help='dataset name: "CelebA", "LFW"')
    parser.add_argument('--target_model_path', type=str, help='path to pretrained target model')
    parser.add_argument("--num_of_identities", type=int, default=None, help="Number of unique identities")
    parser.add_argument("--num_per_identity", type=int, default=None, help="Number of images per identities")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")

    opt = parser.parse_args()

    return opt

def pca(dataset, target_model: str, dataset_name: str, image_num:int, device):
    pkl_path = resolve_path('../../../results/pca', f'{target_model}_{dataset_name}_all_features.pkl')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            all_target_features = pickle.load(f)
    else:
        img_size_T = get_img_size(target_model)
        T, _ = load_model_as_feature_extractor(
            arch=target_model,
            embedding_size=512,
            mode='eval',
            path='',
            pretrained=True
        )
        if isinstance(T, nn.Module):
            T.to(device) 
            T.eval()

        dataloader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            # Optimization:
            num_workers=os.cpu_count(),
            pin_memory=True
        )

        transform = transforms.Compose([
            transforms.Resize((img_size_T, img_size_T)),
        ])

        all_target_features = torch.Tensor([]).to(device)
        for batch, _ in tqdm(dataloader):
            target_features = T(transform(batch.to(device))).detach().to(device)
            all_target_features= torch.cat((all_target_features, target_features))

        with open(pkl_path, 'wb') as f:
            pickle.dump(all_target_features, f)

    all_target_features = all_target_features.cpu()
    pca = PCA()
    df = pd.DataFrame(all_target_features)
    pca.fit(df)
    feature = pca.transform(df)
    features = pd.DataFrame(feature, columns=["PC{}".format(x + 1) for x in range(len(df.columns))])
    features = features.sort_values(by='PC1')
    filter = list(range(1, len(features), int(len(features)/image_num)))
    filter = features.iloc[filter].sort_index().index.to_list()
    return filter

# def set_global():
#     global options
#     global device

#     options = get_options()
#     # Decide device
#     # gpu_id = get_freer_gpu()
#     # device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

#     # Decide device
#     device = f"cuda:{options.gpu_idx}" if torch.cuda.is_available() else "cpu"

#     options.device = device

def main():
    options = get_options()

    # Decide device
    device = f"cuda:{options.gpu_idx}" if torch.cuda.is_available() else "cpu"
    options.device = device

    # Create directories to save data
    # Create directory to save results
    result_dir = resolve_path(options.result_dir, options.identifier)
    os.makedirs(result_dir, exist_ok=True)

    # Save step1 options
    save_json(resolve_path(result_dir, "step1.json"), vars(options))

    # Create logger
    logger = create_logger(f"Step 1", resolve_path(result_dir, "training.log"))

    # Log options
    logger.info(vars(options))

    dataset = CelebA(
        base_dir=options.dataset_dir,
        usage='train',
        select=(options.num_of_identities, options.num_per_identity),
        transform=transforms.ToTensor()
    )

    filter = pca(dataset=dataset, target_model=options.target_model, dataset_name=options.dataset, image_num=10000, device=device)
    print(filter)

if __name__ == '__main__':
    main()