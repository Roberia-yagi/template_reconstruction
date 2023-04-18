import sys
sys.path.append("../")

import os
import argparse
import numpy as np
import itertools
from tqdm import tqdm 
import datetime
from typing import Any, Union, Tuple, Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from util import extract_features_from_nnModule, save_json, load_json, load_model_as_feature_extraction,create_logger, resolve_path
from fairface import Fairface
import umap
import matplotlib.pyplot as plt


def get_options() -> Any:
    parser = argparse.ArgumentParser()

    # Timestamp
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    parser.add_argument("--identifier", type=str, default=time_stamp, help="name of result folder")

    # Folders
    parser.add_argument("--save", type=bool, default=False, help="if save log and data")
    parser.add_argument("--dataset_dir", type=str, default="~/nas/dataset/fairface", help="path to directory which includes dataset(Fairface)")
    parser.add_argument("--result_dir", type=str, default="~/nas/results/attribute_inversion/step1", help="path to directory which includes results")

    # Conditions
    parser.add_argument("--gpu_idx", type=int, default=0, help="index of cuda devices")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--img_channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--data_num", type=int, default=10000, help="number of data to use")
    parser.add_argument("--attack_model", type=str, default="Arcface", help="attack model(feature extraction): 'Arcface', 'FaceNet'")
    parser.add_argument("--target_model", type=str, default="FaceNet", help="target model(feature extraction): 'Arcface', 'FaceNet'")
    parser.add_argument("--attributes", nargs='+', help="attributes to slice the dataset(age, gender, race)")

    opt = parser.parse_args()

    return opt


def load_fairface_dataset_loader(
    base_dir: str,
    usage: str,
    data_num: int,
    attribute_group_list: dict,
    options: Any,
) -> Tuple[Fairface, list]:
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    attribute_list = []

    for attribute_group in attribute_group_list:
        if attribute_group in options.attributes:
            attribute_list.append(attribute_group_list[attribute_group])

    attributes_list = list(itertools.product(*attribute_list))

    datasets = dict()
    dataloaders = dict()

    for attributes in attributes_list:
        dataset = Fairface(
            base_dir=base_dir,
            usage=usage,
            transform=transform,
            data_num=data_num,
            attributes=attributes
        )

        datasets[attributes] = dataset

        dataloader = DataLoader(
            dataset,
            batch_size=options.batch_size,
            shuffle=False,
            # Optimization:
            num_workers=os.cpu_count(),
            pin_memory=True
        )

        dataloaders[attributes] = dataloader

    return datasets, dataloaders, attributes_list

def calculate_average_of_features(attribute_feature: dict,
                              attribute: str) -> torch.Tensor:
    features = attribute_feature[attribute]
    average_of_features = torch.mean(features, dim=0)
    return average_of_features

def register_dist_from_average_L2(attribute_feature: dict,
                               averages_of_features: torch.Tensor,
                               attribute: str,
                               device: Any):
        features = attribute_feature[attribute]

        dist_from_average = torch.cdist(features, averages_of_features, p=2)
        attribute_feature[attribute] = (features, dist_from_average)


def register_dist_from_average_cos_sim(attribute_feature: dict,
                               averages_of_features: torch.Tensor,
                               attribute: str,
                               device: Any):
        features = attribute_feature[attribute]

        metric = nn.CosineSimilarity(dim=1)
        cosine_similarities = torch.Tensor().to(device)
        for averages_for_attribute in averages_of_features:
            cosine_similarity = metric(averages_for_attribute, features)
            cosine_similarities = torch.cat((cosine_similarities, cosine_similarity.unsqueeze(0)), 0)
        attribute_feature[attribute] = (features, cosine_similarities)

def transform_batch_for_extraction(batch: list,
                                   transform: transforms) -> torch.Tensor:
    return transform(batch)


def main():
    options = get_options()

    # Decide device
    device = f"cuda:{options.gpu_idx}" if torch.cuda.is_available() else "cpu"
    options.device = device

    # Create directories to save data
    if options.save:
        # Create directory to save results
        result_dir = resolve_path(options.result_dir, options.identifier)
        os.makedirs(result_dir, exist_ok=True)

        # Save step1 options
        save_json(resolve_path(result_dir, "step1.json"), vars(options))

    # Create logger
    if options.save:
        logger = create_logger(f"Step 1", resolve_path(result_dir, "training.log"))
    else:
        logger = create_logger(f"Step 1")

    # Log options
    logger.info(vars(options))

    # Load attribute list
    attribute_group_list = load_json("attributes.json")

    T, img_size_T, _ = load_model_as_feature_extraction(options.target_model, options)
    F, img_size_F, _ = load_model_as_feature_extraction(options.attack_model, options)
    if isinstance(T, nn.Module):
        T.to(device) 
        T.eval()
    if isinstance(F, nn.Module):
        F.to(device) 
        F.eval()

    _, dataloaders, _= load_fairface_dataset_loader(options.dataset_dir, \
                                                'train', \
                                                options.data_num, \
                                                attribute_group_list, \
                                                options)


    transform_T = transforms.Compose([
        transforms.Resize((img_size_T, img_size_T)),
    ])
    transform_F = transforms.Compose([
        transforms.Resize((img_size_F, img_size_F)),
    ])

    attribute_feature_T = dict()
    attribute_feature_F = dict()
    for dataloader_name in tqdm(dataloaders):
        features_T = torch.Tensor().to(device)
        features_F = torch.Tensor().to(device)
        for batch in dataloaders[dataloader_name]:
            batch_T = transform_batch_for_extraction(batch[0], transform_T) 
            batch_F = transform_batch_for_extraction(batch[0], transform_F) 
            feature_T = extract_features_from_nnModule(batch_T,  T, None, device)
            feature_F = extract_features_from_nnModule(batch_F,  F, None, device)
            features_T = torch.cat((features_T, feature_T.detach()), 0)
            features_F = torch.cat((features_F, feature_F.detach()), 0)
        attribute_feature_T[dataloader_name] = features_T
        attribute_feature_F[dataloader_name] = features_F


    metric = nn.CosineSimilarity(dim=1)

    # The ratio of the distance of a man to all men and to women, and vice versa
    # Facenet Eucrid base
    man= 0
    woman= 0
    for i in tqdm(range(len(attribute_feature_T[('Male',)]))):
        Male_attribute_for_male = attribute_feature_T[('Male',)][i].expand(len(attribute_feature_T[('Male',)]), -1)
        Male_attribute_for_female = attribute_feature_T[('Female',)][i].expand(len(attribute_feature_T[('Female',)]), -1)

        a= torch.mean(torch.cdist(Male_attribute_for_male, attribute_feature_T[('Male',)])[0])
        b=torch.mean(torch.cdist(Male_attribute_for_male, attribute_feature_T[('Female',)])[0])
        c=torch.mean(torch.cdist(Male_attribute_for_female, attribute_feature_T[('Male',)])[0])
        d=torch.mean(torch.cdist(Male_attribute_for_female, attribute_feature_T[('Female',)])[0])
        if a < b:
            man += 1
        if d < c:
            woman += 1
    print(man)
    print(woman)

    #Facenet Cos base
    man= 0
    woman= 0
    for i in tqdm(range(len(attribute_feature_T[('Male',)]))):
        Male_attribute_for_male = attribute_feature_T[('Male',)][i].expand(len(attribute_feature_T[('Male',)]), -1)
        Male_attribute_for_female = attribute_feature_T[('Female',)][i].expand(len(attribute_feature_T[('Female',)]), -1)

        a= torch.mean(metric(Male_attribute_for_male, attribute_feature_T[('Male',)]))
        b=torch.mean(metric(Male_attribute_for_male, attribute_feature_T[('Female',)]))
        c=torch.mean(metric(Male_attribute_for_female, attribute_feature_T[('Male',)]))
        d=torch.mean(metric(Male_attribute_for_female, attribute_feature_T[('Female',)]))
        if a > b:
            man += 1
        if d > c:
            woman += 1
    print(man)
    print(woman)

    # Resnet Eucrid base
    man= 0
    woman= 0
    for i in tqdm(range(len(attribute_feature_F[('Male',)]))):
        Male_attribute_for_male = attribute_feature_F[('Male',)][i].expand(len(attribute_feature_F[('Male',)]), -1)
        Male_attribute_for_female = attribute_feature_F[('Female',)][i].expand(len(attribute_feature_F[('Female',)]), -1)

        a= torch.mean(torch.cdist(Male_attribute_for_male, attribute_feature_F[('Male',)])[0])
        b=torch.mean(torch.cdist(Male_attribute_for_male, attribute_feature_F[('Female',)])[0])
        c=torch.mean(torch.cdist(Male_attribute_for_female, attribute_feature_F[('Male',)])[0])
        d=torch.mean(torch.cdist(Male_attribute_for_female, attribute_feature_F[('Female',)])[0])
        if a < b:
            man += 1
        if d < c:
            woman += 1
    print(man)
    print(woman)
    


    # Resnet Eucrid base
    man= 0
    woman= 0
    for i in tqdm(range(len(attribute_feature_F[('Male',)]))):
        Male_attribute_for_male = attribute_feature_F[('Male',)][i].expand(len(attribute_feature_F[('Male',)]), -1)
        Male_attribute_for_female = attribute_feature_F[('Female',)][i].expand(len(attribute_feature_F[('Female',)]), -1)

        a=torch.mean(metric(Male_attribute_for_male, attribute_feature_F[('Male',)]))
        b=torch.mean(metric(Male_attribute_for_male, attribute_feature_F[('Female',)]))
        c=torch.mean(metric(Male_attribute_for_female, attribute_feature_F[('Male',)]))
        d=torch.mean(metric(Male_attribute_for_female, attribute_feature_F[('Female',)]))
        if a > b:
            man += 1
        if d > c:
            woman += 1
    print(man)
    print(woman)


    averages_T = torch.Tensor().to(device)
    averages_F = torch.Tensor().to(device)
    for attribute in attribute_feature_T: # or F
        average_of_features_T = calculate_average_of_features(attribute_feature_T, attribute)
        average_of_features_F = calculate_average_of_features(attribute_feature_F, attribute)
        averages_T = torch.cat((averages_T, average_of_features_T.unsqueeze(0)), 0)
        averages_F = torch.cat((averages_F, average_of_features_F.unsqueeze(0)), 0)
        # print(attribute)
        # print(averages_T.shape)


    for attribute in attribute_feature_T: # or F
        register_dist_from_average_cos_sim(attribute_feature_T, averages_T, attribute, device)
        register_dist_from_average_cos_sim(attribute_feature_F, averages_F, attribute, device)

        # print(attribute)
        # print("T for male:", torch.mean(attribute_feature_T[attribute][1][0]))
        # print("T for female:", torch.mean(attribute_feature_T[attribute][1][1]))
        # print("F for male:", torch.mean(attribute_feature_F[attribute][1][0]))
        # print("F for female:", torch.mean(attribute_feature_F[attribute][1][1]))


if __name__ == '__main__':
    main()
