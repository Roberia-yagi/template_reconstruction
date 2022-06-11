# Invert feature to image with converter trained in step 1
import os
import sys
sys.path.append("../")
sys.path.append("../../")
import argparse
import datetime
from typing import Any, Tuple

import torch
import pickle
from tqdm import tqdm
from torch import nn
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.util import (save_json, load_json, create_logger, resolve_path, load_model_as_feature_extractor,
    get_img_size, load_autoencoder)
from utils.email_sender import send_email
from utils.fairface import Fairface
from utils.lfw import LFW
import matplotlib
import matplotlib.pyplot as plt

def normalize_hist(x, p):
    max = 0
    for item in p:
        height = item.get_height()/sum(x)
        item.set_height(height)
        if height > max:
            max = height
    return max

def get_options() -> Any:
    parser = argparse.ArgumentParser()

    # Timestamp
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    parser.add_argument("--identifier", type=str, default=time_stamp, help="timestamp")
    parser.add_argument("--gpu_idx", type=int, default=0, help="index of cuda devices")

    # Save
    parser.add_argument("--save", type=bool, default=True, help="if save log and data")

    # Directories
    parser.add_argument("--result_dir", type=str, default="~/nas/results/show_transferer_acc_with_hist", help="path to directory which includes results")
    parser.add_argument("--step1_dirs", nargs='+', required=True, help="path to directory which includes the step1 result")
    parser.add_argument("--same_pickle_path", type=str, default='/home/akasaka/nas/results/show_model_acc_with_hist/2022_06_05_04_34_Original_with_pickle/FaceNet_same_cossim.pkl',
                        help="path to directory which includes the pickle of original data")
    parser.add_argument("--diff_pickle_path", type=str, default='/home/akasaka/nas/results/show_model_acc_with_hist/2022_06_05_04_34_Original_with_pickle/FaceNet_diff_cossim.pkl',
                        help="path to directory which includes the pickle of original data")
    parser.add_argument("--target_image_dir", type=str, default="~/nas/dataset/fairface/train", help="path to directory which contains target images")
    parser.add_argument('--target_model_path', default='',
                        type=str, help='path to pretrained target model')
    parser.add_argument('--attack_model_path', default='',
                        type=str, help='path to pretrained attack model')

    # For inference
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")

    # Conditions
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--target_model", type=str, default="Magface", help="target model: 'FaceNet', 'Arcface', 'Magface")
    parser.add_argument("--attack_model", type=str, default="FaceNet", help="attack model: 'FaceNet', 'Arcface', 'Magface")
    parser.add_argument("--AE_ver", type=int, default=1.1, help="AE version: 1, 1.1, 1.2, 1.3, 1.4")
    parser.add_argument("--transferer_training_epoch", type=int, default=-1, help="dimensionality of the latent space")
    parser.add_argument("--embedding_size", type=int, default=512, help="embedding size of features of target model:[128, 512]")

    opt = parser.parse_args()

    return opt

def get_best_image(FE: nn.Module, images: nn.Module, image_size: int, all_target_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    resize = transforms.Resize((image_size, image_size))
    metric = nn.CosineSimilarity(dim=1)
    FEz_features = FE(resize(images))
    dim = FEz_features.shape[1]
    sum_of_cosine_similarity = 0
    for target_feature in all_target_features.view(-1, dim):
        target_feature = target_feature.expand(FEz_features.shape[0], -1)
        sum_of_cosine_similarity += metric(FEz_features, target_feature)
    sum_of_cosine_similarity /= all_target_features.shape[0]
    bestImageIndex = sum_of_cosine_similarity.argmax()
    return images[bestImageIndex], sum_of_cosine_similarity[bestImageIndex]


def set_global():
    global options
    global device
    global metric
    options = get_options()
    metric = nn.CosineSimilarity(dim=1)
    # Decide device
    # gpu_id = get_freer_gpu()
    # device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    device = f"cuda:{options.gpu_idx}" if torch.cuda.is_available() else "cpu"

    options.device = device

def load_transferer(step1_dir: str, ver: int):
    if options.transferer_training_epoch > -1:
        model_path = resolve_path(step1_dir, f'AE_{options.attack_model_epochs}.pth') 
    else:
        model_path = resolve_path(step1_dir, 'AE.pth') 
    C = load_autoencoder(
        model_path=model_path,
        pretrained=True,
        mode='eval',
        ver=ver
    ).to(device)
    
    if isinstance(C, nn.Module):
        C.to(device) 
        C.eval()
    return C

def main():
    set_global()
    if options.save:
        # Create directory to save results
        result_dir = resolve_path(options.result_dir, options.identifier)
        os.makedirs(result_dir, exist_ok=True)

        # Save options by json format
        save_json(resolve_path(result_dir, "step2.json"), vars(options))

    # Create logger
    if options.save:
        logger = create_logger(f"Step 2", resolve_path(result_dir, "inference.log"))
    else:
        logger = create_logger(f"Step 2")

    # Log options
    logger.info(vars(options))

    # Load models
    img_size_T = get_img_size(options.target_model)
    img_size_A = get_img_size(options.attack_model)

    T, _ = load_model_as_feature_extractor(
        arch=options.target_model,
        embedding_size=options.embedding_size,
        mode='eval',
        path=options.target_model_path,\
        pretrained=True
    )
    A, _ = load_model_as_feature_extractor(
        arch=options.attack_model,
        embedding_size=options.embedding_size,
        mode='eval',
        path=options.attack_model_path,
        pretrained=True
    )
    if isinstance(T, nn.Module):
        T.to(device) 
        T.eval()
    if isinstance(A, nn.Module):
        A.to(device) 
        A.eval()

    transform_T = transforms.Compose([
        transforms.Resize((img_size_T, img_size_T)),
    ])
    transform_A = transforms.Compose([
        transforms.Resize((img_size_A, img_size_A)),
    ])



    # Load datasets
    dataset = LFW(
        base_dir='../../../dataset/LFWA/lfw-deepfunneled-MTCNN160',
        # usage='train',
        # data_num=None,
        # attributes=None,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )
    print(len(dataset))

    random_dataloader = DataLoader(
        dataset,
        batch_size=options.batch_size,
        shuffle=True,
        # Optimization:
        num_workers=os.cpu_count(),
        pin_memory=True
    )

    # Calculate cossim of all combination of datas of same person
    cossims = torch.Tensor().to(device)
    max__ = 0
    for step1_dir, color in zip(options.step1_dirs, matplotlib.colors.BASE_COLORS):
        C = load_transferer(step1_dir, options.AE_ver)
        for batch_data, id in tqdm(random_dataloader):
            converted_target_image_T = transform_T(batch_data).to(device)
            converted_target_image_A = transform_A(batch_data).to(device)

            target_feature_in_T = T(converted_target_image_T.view(batch_data.shape[0], 3, img_size_T, img_size_T)).detach().to(device)
            target_feature_in_A = A(converted_target_image_A.view(batch_data.shape[0], 3, img_size_A, img_size_A)).detach().to(device)
            target_feature_in_A_through_C = C(target_feature_in_T).to(device)
            cossims = torch.concat((cossims, metric(target_feature_in_A, target_feature_in_A_through_C)))

        # Create histogram
        x, _, p = plt.hist(cossims.cpu().numpy(),  bins=50, alpha=0.3, color=color, edgecolor='k', label=step1_dir)
        max_ = normalize_hist(x, p)
        if max_ > max__:
            max__ = max_

    # save data as pickle
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
    plt.savefig(resolve_path(result_dir, f'{options.attack_model}_{options.embedding_size}_hist.png'), bbox_inches='tight')
    plt.clf()

if __name__ == '__main__':
	main()