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

def get_options() -> Any:
    parser = argparse.ArgumentParser()


    # Directories
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    parser.add_argument("--identifier", type=str, default=time_stamp, help="timestamp")
    parser.add_argument("--result_dir", type=str, default="~/nas/results/show_model_acc_with_hist",
                        help="path to directory which includes results")
    parser.add_argument("--dataset_dir", type=str, default="~/nas/dataset/CelebA_MTCNN160", help="path to directory which includes dataset(Fairface)")


    # System preferences
    parser.add_argument("--gpu_idx", type=int, default=0, help="index of cuda devices")

    # Model Params
    parser.add_argument("--embedding_size", type=int, default=512,
                        help="embedding size of features of target model:[128, 512]")
    parser.add_argument("--target_model", type=str, default="FaceNet", 
                        help="target model: 'FaceNet', 'Arcface', 'Magface'")
    parser.add_argument("--target_model_path", type=str, help="path to pth of target model")
    parser.add_argument("--target_model_pretrained", action='store_false', help="path to pth of target model")

    # Runinng Preferences
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")

    opt = parser.parse_args()

    return opt

def calculate_cossim_of_all_combinations(datas: torch.Tensor, criterion: torch.nn):
    size = datas.size(dim=0)
    idxes = torch.combinations(torch.tensor(list(range(size))))

    cossims = torch.Tensor()
    for idx1, idx2 in idxes:
        cossim = criterion(datas[idx1], datas[idx2]).unsqueeze(0).cpu()
        cossims = torch.concat((cossims, cossim))

    return cossims

def normalize_hist(x, p):
    max = 0
    for item in p:
        height = item.get_height()/sum(x)
        item.set_height(height)
        if height > max:
            max = height
    return max


def set_global():
    global options
    global device

    options = get_options()
    # Decide device
    device = f"cuda:{options.gpu_idx}" if torch.cuda.is_available() else "cpu"

    # gpu_id = get_freer_gpu()
    # device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    options.device = device

def main():
    set_global()

if __name__ == '__main__':
    main()

    # Create directory to save results
    result_dir = resolve_path(options.result_dir, options.identifier)
    os.makedirs(result_dir, exist_ok=True)
    
    # Save options by json format
    save_json(resolve_path(result_dir, "step1.json"), vars(options))

    # Create logger
    logger = create_logger(f"Show model acc with histogram", resolve_path(result_dir, "training.log"))

    # Log options
    logger.info(vars(options))

    # Load models
    img_size = get_img_size(options.target_model)

    model, params_to_update = load_model_as_feature_extractor(
        arch=options.target_model,
        embedding_size=options.embedding_size,
        mode='eval',
        path=options.target_model_path,
        pretrained=options.target_model_pretrained
    )
    model.to(device)

    # Load datasets
    dataset = CelebA(
        base_dir=options.dataset_dir,
        usage='all',
        transform=transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]),
        sorted=True
    )

    random_dataloader = DataLoader(
        dataset,
        batch_size=options.batch_size,
        shuffle=True,
        # Optimization:
        num_workers=os.cpu_count(),
        pin_memory=True
    )

    criterion = nn.CosineSimilarity(dim=0)

    # Calculate cossim of all combination of datas of same person
    current_id = None
    same_datas= torch.Tensor()
    same_cossims = torch.Tensor()
    count = 0
    transform = transforms.ToPILImage()
    for i, (data, id) in enumerate(tqdm(dataset)):
        # if i == 1000:
        #     break
        data = data.unsqueeze(0).to(device)
        if current_id is None or current_id != id:
            if same_datas.size(dim=0) != 0:
                same_features = model(same_datas)
                cossims = calculate_cossim_of_all_combinations(same_features, criterion)
                same_cossims = torch.concat((same_cossims, cossims))
            same_datas = data
            current_id = id
        elif current_id == id:
            same_datas= torch.concat((same_datas, data))
        
    # Sample datas of different person rondamly and calculate
    dif_cossims = torch.Tensor()
    done = False
    while not done:
        for batch_data, id in tqdm(random_dataloader):
            sorted_id, _ = torch.sort(id)

            if dif_cossims.size(dim=0) > same_cossims.size(dim=0):
                done = True
                break
            if torch.equal(torch.unique(id), sorted_id):
                batch_data = batch_data.to(device)
                dif_features = model(batch_data)
                if dif_features.size(dim=0) == options.batch_size:
                    cossims = criterion(dif_features[:int(options.batch_size/2)],
                                        dif_features[int(options.batch_size/2):]).cpu()
                    dif_cossims = torch.concat((dif_cossims, cossims))

    # Create histogram
    x1, _, p1 = plt.hist(same_cossims.numpy(),  color='green', bins=28, alpha=0.3, label='Same')
    x2, _, p2 = plt.hist(dif_cossims.numpy(), color='blue', bins=28, alpha=0.3, label='Diff')
    plt.xlabel("Cos sim")
    plt.ylabel("Freq")
    plt.legend()
    max1 = normalize_hist(x1, p1)
    max2 = normalize_hist(x2, p2)
    plt.ylim(0, max(max1, max2))
    plt.savefig(resolve_path(result_dir, f'{options.target_model}_{options.embedding_size}_hist.png'))
    plt.clf()

    # save data as pickle
    with open(resolve_path(result_dir, f'{options.target_model}_same_cossim.pkl'), 'wb') as f:
        pickle.dump(same_cossims.numpy(), f)
    with open(resolve_path(result_dir, f'{options.target_model}_diff_cossim.pkl'), 'wb') as f:
        pickle.dump(dif_cossims.numpy(), f)

    # Calculate scores and draw ROC curve
    x = np.r_[same_cossims.numpy(), dif_cossims.numpy()]
    y = np.r_[np.ones(same_cossims.numpy().shape), np.zeros(dif_cossims.numpy().shape)]
    fpr, tpr, thresholds = roc_curve(y, x)

    threshold_idx = np.argmin(fpr - tpr)
    logger.info(f"[Threshold: {thresholds[threshold_idx]}] [FPR: {fpr[threshold_idx]}] [TPR: {tpr[threshold_idx]}]")
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.savefig(resolve_path(result_dir, f'{options.target_model}_{options.embedding_size}_ROC.png'))
