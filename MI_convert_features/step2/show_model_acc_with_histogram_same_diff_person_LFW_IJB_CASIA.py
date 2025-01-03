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
import json

from typing import Any, List, Tuple

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
from utils.lfw import LFW
from utils.ijb import IJB
from utils.casia_web_face import CasiaWebFace
from torch import nn
import matplotlib.pyplot as plt

from utils.util import (save_json,  create_logger, resolve_path, load_model_as_feature_extractor, 
load_model_as_feature_extractor, set_global)

def get_options() -> Any:
    parser = argparse.ArgumentParser()


    # Directories
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    parser.add_argument("--identifier", type=str, default=time_stamp, help="timestamp")
    parser.add_argument("--result_dir", type=str, default="../../../results/show_model_acc_with_hist_LFW",
                        help="path to directory which includes results")
    parser.add_argument("--dataset", type=str, required=True, help="test dataset:[LFW, IJB-C, CASIA]")
    parser.add_argument("--dataset_dir", type=str, required=True, help="path to the dataset directory")


    # System preferences
    parser.add_argument("--gpu_idx", type=int, default=None, help="index of cuda devices")

    # Model Params
    parser.add_argument("--embedding_size", type=int, default=512,
                        help="embedding size of features of target model:[128, 512]")
    parser.add_argument("--target_model", type=str, required=True, 
                        help="target model: 'FaceNet', 'Arcface', 'Magface'")
    parser.add_argument("--target_model_path", type=str, help="path to pth of target model")
    parser.add_argument("--target_model_pretrained", action='store_false', help="path to pth of target model")

    # Runinng Preferences
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")

    opt = parser.parse_args()

    return opt

def calculate_cossim_of_all_combinations(datas: torch.Tensor, criterion: torch.nn, filenames: List) -> Tuple[torch.Tensor, List]:
    size = datas.size(dim=0)
    idxes = torch.combinations(torch.tensor(list(range(size))))

    filenames_return = []
    cossims = torch.Tensor()
    for idx1, idx2 in idxes:
        cossim = criterion(datas[idx1], datas[idx2]).unsqueeze(0).cpu()
        filename = filenames[idx1] + '_' + filenames[idx2]
        cossims = torch.concat((cossims, cossim))
        filenames_return.append(filename)

    return cossims, filenames_return

def normalize_hist(x, p):
    max = 0
    for item in p:
        height = item.get_height()/sum(x)
        item.set_height(height)
        if height > max:
            max = height
    return max


# def fixed_image_standardization(image_tensor):
#     processed_tensor = (image_tensor - 127.5) / 128.0
#     return processed_tensor

def create_threshold_dict(dict, name, idx, thresholds, fpr, tpr):
    dict[name]['threshold'] = str(thresholds[idx])
    dict[name]['fpr'] = str(fpr[idx])
    dict[name]['tpr'] = str(tpr[idx])

    return dict
    


def main():
    global options
    global device
    device, options = set_global(get_options)

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
    model, _ = load_model_as_feature_extractor(
        arch=options.target_model,
        embedding_size=options.embedding_size,
        mode='eval',
        path=options.target_model_path,
        pretrained=options.target_model_pretrained
    )
    model.to(device)


    opencv = False
    if options.target_model == 'Magface':
        opencv = True

    # Load datasets
    if options.dataset == 'LFW':
        dataset = LFW(
            base_dir=options.dataset_dir,
            transform=transforms.Compose([
                transforms.ToTensor(),
                # fixed_image_standardization
            ]),
            opencv=opencv
        )
    elif options.dataset == 'IJB-C':
        dataset = IJB(
            base_dir=options.dataset_dir,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]),
            opencv=opencv
        )
    elif options.dataset == 'CASIA':
        dataset = CasiaWebFace(
            base_dir=options.dataset_dir,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]),
            num_of_identities=5120,
            num_per_identity=20,
            usage='train',
            # opencv=opencv
        )
    else:
        raise('dataset is invalid')

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
    # Filename: for debug
    current_id = None
    same_filenames = []
    same_datas= torch.Tensor()
    same_cossims = torch.Tensor()
    for i, (data, (id, filename)) in enumerate(tqdm(dataset)):
        # if i > 100:
        #     break
        data = data.unsqueeze(0).to(device)
        if current_id is None or current_id != id:
            if same_datas.size(dim=0) != 0:
                same_features = model(same_datas)
                cossims, filenames = calculate_cossim_of_all_combinations(same_features, criterion, same_filenames)
                same_cossims = torch.concat((same_cossims, cossims))
                # for cossim, filename in zip(cossims, filenames):
                #     if cossim < 0.1:
                #         print(cossim)
                #         print(filename)
            same_datas = data
            same_filenames = [filename]
            current_id = id
        elif current_id == id:
            same_datas= torch.concat((same_datas, data))
            same_filenames.append(filename)

    print(same_cossims.size())
        
    # Sample datas of different person rondamly and calculate
    # if the length of set_id equals that of id, the identities in batch_data do not overlap.
    dif_cossims = torch.Tensor()
    done = False
    while not done:
        for batch_data, (id, _) in tqdm(random_dataloader):
            set_id = set(id)
            if dif_cossims.size(dim=0) > same_cossims.size(dim=0):
                done = True
                break
            if len(set_id) == len(id):
                batch_data = batch_data.to(device)
                dif_features = model(batch_data)
                if dif_features.size(dim=0) == options.batch_size:
                    cossims = criterion(dif_features[:int(options.batch_size/2)],
                                        dif_features[int(options.batch_size/2):]).cpu()
                    dif_cossims = torch.concat((dif_cossims, cossims))

    # Create a histogram
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

    # save data as pickles
    with open(resolve_path(result_dir, f'same.pkl'), 'wb') as f:
        pickle.dump(same_cossims.numpy(), f)
    with open(resolve_path(result_dir, f'diff.pkl'), 'wb') as f:
        pickle.dump(dif_cossims.numpy(), f)

    # Calculate scores and draw ROC curve
    x = np.r_[same_cossims.numpy(), dif_cossims.numpy()]
    y = np.r_[np.ones(same_cossims.numpy().shape), np.zeros(dif_cossims.numpy().shape)]
    fpr, tpr, thresholds = roc_curve(y, x, drop_intermediate=False)

    results_threshold = {'EER': {}, 'fpr1': {}, 'fpr0.1': {}, 'fpr0.01': {}}
    threshold_EER_idx = np.argmin(np.abs(1 - fpr - tpr))
    logger.info(f"[Threshold at EER: {thresholds[threshold_EER_idx]}] [FPR: {fpr[threshold_EER_idx]}] [TPR: {tpr[threshold_EER_idx]}]")
    create_threshold_dict(results_threshold, 'EER', threshold_EER_idx, thresholds, fpr, tpr)

    threshold_1_idx = np.argmin(np.abs(fpr - .01))
    logger.info(f"[Threshold at FPR 1%: {thresholds[threshold_1_idx]}] [FPR: {fpr[threshold_1_idx]}] [TPR: {tpr[threshold_1_idx]}]")
    create_threshold_dict(results_threshold, 'fpr1', threshold_1_idx, thresholds, fpr, tpr)

    threshold_point_1_idx = np.argmin(np.abs(fpr - .001))
    logger.info(f"[Threshold at FPR 0.1%: {thresholds[threshold_point_1_idx]}] [FPR: {fpr[threshold_point_1_idx]}] [TPR: {tpr[threshold_point_1_idx]}]")
    create_threshold_dict(results_threshold, 'fpr0.1', threshold_point_1_idx, thresholds, fpr, tpr)

    threshold_point_01_idx = np.argmin(np.abs(fpr - .0001))
    logger.info(f"[Threshold at FPR 0.01%: {thresholds[threshold_point_01_idx]}] [FPR: {fpr[threshold_point_01_idx]}] [TPR: {tpr[threshold_point_01_idx]}]")
    create_threshold_dict(results_threshold, 'fpr0.01', threshold_point_01_idx, thresholds, fpr, tpr)

    with open(resolve_path(result_dir, './thresholds.json'), 'w') as f:
        json.dump(results_threshold, f)

    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.savefig(resolve_path(result_dir, f'{options.target_model}_{options.embedding_size}_ROC.png'))

if __name__ == '__main__':
    main()