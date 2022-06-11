import sys
sys.path.append("../")
sys.path.append("../../")

import os
import argparse
import itertools
from tqdm import tqdm 
import datetime
from typing import Any, Tuple, Dict, List
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from utils.util import (extract_features_from_nnModule, save_json, load_json, load_model_as_feature_extraction,create_logger, resolve_path, get_output_shape)
from utils.fairface import Fairface
from model_original import Autoencoder


def get_options() -> Any:
    parser = argparse.ArgumentParser()

    # Timestamp
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    parser.add_argument("--identifier", type=str, default=time_stamp, help="name of result folder")

    # Folders
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
    parser.add_argument("--attributes", nargs='+', default="gender", help="attributes to slice the dataset(age, gender, race)")

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

def calculate_average_of_features(attribute_feature: dict, attribute: str) -> torch.Tensor:
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


def register_dist_from_average_cos_sim(attribute_to_feature: dict,
                               attribute_to_average: dict(),
                               attribute_to_distance_dict: dict(),
                               attribute_from: str,
                               device: Any):

        attribute_to_distance_from_average = dict()
        features = attribute_to_feature[attribute_from]

        metric = nn.CosineSimilarity(dim=1)
        for attribute_to in attribute_to_average:
            average = attribute_to_average[attribute_to]
            attribute_to_distance_from_average[attribute_to] = metric(average, features)
        attribute_to_distance_dict[attribute_from] = attribute_to_distance_from_average

def transform_batch_for_extraction(batch: list,
                                   transform: transforms) -> torch.Tensor:
    return transform(batch)

def save_graph(data: dict(), base_attribute: tuple, graph_name: str, options:Any):
    columns = []
    values = np.empty(0)
    for key in data.keys():
        columns.append(disband_tuple(key))
        values = np.append(values, torch.mean(data[base_attribute][key]).cpu().numpy())
    plt.bar(height=values, x=columns)
    plt.ylim(bottom=values.min(), top=values.max())
    plt.xticks(rotation=90)
    plt.savefig(resolve_path(options.result_dir, options.identifier, graph_name), bbox_inches='tight')
    plt.clf()

def disband_tuple(tuple: tuple):
    return ''.join(tuple)

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

    # Load attribute list
    attribute_group_list = load_json("attributes.json")

    # Load models
    T, img_size_T, _ = load_model_as_feature_extraction(options.target_model, options)
    F, img_size_F, _ = load_model_as_feature_extraction(options.attack_model, options)
    # AE = Autoencoder(feature_size=get_output_shape(T, img_size_T))
    if isinstance(T, nn.Module):
        T.to(device) 
        T.eval()
    if isinstance(F, nn.Module):
        F.to(device) 
        F.eval()
    # AE.to(device)

    # Load datasets
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

    # 
    

    # Extract features from model
    attribute_to_feature_T = dict()
    attribute_to_feature_F = dict()
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
        attribute_to_feature_T[dataloader_name] = features_T
        attribute_to_feature_F[dataloader_name] = features_F


    # Calculate averages of features
    # attribute_to_distance_dict[attribute][attribute] = distance (distance from all features of first attribute to the average feature of second attribute)
    attribute_to_average_T = dict()
    attribute_to_average_F = dict()
    for attribute in attribute_to_feature_T: # or F
        average_of_features_T = calculate_average_of_features(attribute_to_feature_T, attribute)
        average_of_features_F = calculate_average_of_features(attribute_to_feature_F, attribute)
        attribute_to_average_T[attribute] = average_of_features_T
        attribute_to_average_F[attribute] = average_of_features_F

    attribute_to_distance_dict_T = dict()
    attribute_to_distance_dict_F = dict()
    for attribute_from in attribute_to_feature_T:
        register_dist_from_average_cos_sim(attribute_to_feature_T, attribute_to_average_T, attribute_to_distance_dict_T, attribute_from, device)

        for attribute_to in attribute_to_feature_T:
            print(f"cos sim from average of {attribute_from} to {attribute_to} in T is {torch.mean(attribute_to_distance_dict_T[attribute_from][attribute_to])}")

    for attribute_from in attribute_to_feature_F:
        register_dist_from_average_cos_sim(attribute_to_feature_F, attribute_to_average_F, attribute_to_distance_dict_F, attribute_from, device)

        for attribute_to in attribute_to_feature_F:
            print(f"cos sim from average of {attribute_from} to {attribute_to} in F is {torch.mean(attribute_to_distance_dict_F[attribute_from][attribute_to])}")

    for attribute_tuple in attribute_to_distance_dict_T:
        attribute_str = disband_tuple(attribute_tuple)
        save_graph(attribute_to_distance_dict_T, attribute_tuple, f'{attribute_str}_T_graph.png', options)
        save_graph(attribute_to_distance_dict_F, attribute_tuple, f'{attribute_str}_F_graph.png', options)

if __name__ == '__main__':
    main()
