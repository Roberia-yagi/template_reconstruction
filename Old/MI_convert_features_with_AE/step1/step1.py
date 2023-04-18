# Train AE
import sys

from sklearn import datasets

sys.path.append("../")
sys.path.append("../../")

import os
import itertools
import argparse
import pickle
import time
import datetime
from tqdm import tqdm 
from typing import Any, List, Tuple

from util.models.AutoEncoder import AutoEncoder

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from util.fairface import Fairface

from util.util import (save_json, load_json, create_logger, resolve_path, load_model_as_feature_extractor, BatchLoader,
    get_freer_gpu, get_img_size, extract_features_from_nnModule, get_output_shape)

def get_options() -> Any:
    parser = argparse.ArgumentParser()

    # Timestamp
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    parser.add_argument("--identifier", type=str, default=time_stamp, help="timestamp")

    # Save
    parser.add_argument("--save", type=bool, default=True, help="if save log and data")

    # Directories
    parser.add_argument("--result_dir", type=str, default="~/nas/results/MI_convert_features_with_AE/step1", help="path to directory which includes results")
    parser.add_argument("--dataset_dir", type=str, default="~/nas/dataset/fairface", help="path to directory which includes dataset(Fairface)")

    # Conditions
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--target_model", type=str, default="Arcface", help="target model: 'FaceNet', 'Arcface'")
    parser.add_argument("--learning_rate", type=float, default=1e-1, help="learning rate of optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-8, help="weight decay of optimizer")
    parser.add_argument("--attributes", nargs='+', default="gender", help="attributes to slice the dataset(age, gender, race)")
    parser.add_argument("--data_num", type=int, default=1000000, help="number of data to use")

    opt = parser.parse_args()

    return opt

def calculate_average_of_features(attribute_feature: dict, attribute: str) -> torch.Tensor:
    features = attribute_feature[attribute]
    average_of_features = torch.mean(features, dim=0)
    return average_of_features

def transform_batch_for_extraction(batch: list,
                                   transform: transforms) -> torch.Tensor:
    return transform(batch)

def load_fairface_dataset_loader_by_attribute(
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

def load_fairface_dataset_loader_for_all(
    base_dir: str,
    usage: str,
    data_num: int,
    options: Any,
) -> Tuple[Fairface, list]:
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

def convert_dict_to_tensor(dict: dict):
    l = list(dict.values())
    converted_tensor = torch.Tensor().to(device)
    for e in l:
        converted_tensor = torch.cat((converted_tensor, e.unsqueeze(0)), 0)
    
    # print(type(converted_tensor))
    return converted_tensor

def calculate_dist_from_batch_to_average(batch: torch.Tensor, averages: torch.Tensor, model: nn.modules):
    batch = batch.to(device)
    features = model(batch)
    dist_from_batch_to_average = torch.Tensor().to(device)
    for feature in features:
        dist = metric(averages, feature)
        dist_from_batch_to_average = torch.cat((dist_from_batch_to_average, dist.unsqueeze(0)), 0)
    
    return dist_from_batch_to_average


def create_feature_to_dist_from_average(dataloaders_by_attribute: dict, dataloader_for_all: dict, transform: transforms, model: nn.modules):
    # Extract features from model
    attribute_to_feature = dict()
    for dataloader_name in tqdm(dataloaders_by_attribute):
        features = torch.Tensor().to(device)
        for batch in dataloaders_by_attribute[dataloader_name]:
            batch = transform_batch_for_extraction(batch[0], transform) 
            feature = extract_features_from_nnModule(batch,  model, None, device)
            features = torch.cat((features, feature.detach()), 0)
        attribute_to_feature[dataloader_name] = features

    # Calculate averages of features
    # attribute_to_distance_dict[attribute][attribute] = distance (distance from all features of first attribute to the average feature of second attribute)
    attribute_to_average = dict()
    for attribute in attribute_to_feature: # or A
        average_of_features = calculate_average_of_features(attribute_to_feature, attribute)
        attribute_to_average[attribute] = average_of_features


    dist_from_features_to_averages = torch.Tensor().to(device)
    averages = convert_dict_to_tensor(attribute_to_average)
    for batch, _ in dataloader_for_all:
        batch = transform_batch_for_extraction(batch, transform) 
        dist_from_batch_to_average = calculate_dist_from_batch_to_average(batch, averages, model)
        dist_from_features_to_averages = torch.cat((dist_from_features_to_averages, dist_from_batch_to_average), 0)

    return dist_from_features_to_averages, averages

def load_feature_to_dist_from_average(dataloaders_by_attribute: dict, dataset_for_all: datasets, transform: transforms, model: nn.modules, name: str):
    pickle_path = resolve_path(options.dataset_dir, ''.join(options.attributes)) +  name + ".pkl"
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            feature_to_average, averages = pickle.load(f)
            print("Loaded features from pickle")
    else:
        feature_to_average, averages = create_feature_to_dist_from_average(dataloaders_by_attribute, dataset_for_all, transform, model)
        with open(pickle_path, "wb") as f:
            pickle.dump((feature_to_average, averages), f)
            print("Saved features as pickle")

    return feature_to_average, averages

def calculate_dist_from_feature_to_average(attribute_to_feature: dict,
                               attribute_to_average: dict(),
                               attribute_to_distance_dict: dict(),
                               attribute_from: str,
                               device: Any):

        attribute_to_distance_from_average = dict()
        features = attribute_to_feature[attribute_from]

        for attribute_to in attribute_to_average:
            average = attribute_to_average[attribute_to]
            attribute_to_distance_from_average[attribute_to] = metric(average, features)
        attribute_to_distance_dict[attribute_from] = attribute_to_distance_from_average

# def loss_func(A, B):
#     print(A.shape)
#     print(type(A))
#     print(B.shape)
#     print(type(B))
#     return nn.MSELoss(A, B)

# def loss_func(dists_A, dists_B):
#     return dists_A - dists_B

def set_global():
    global options
    global device
    global metric

    options = get_options()
    metric = nn.CosineSimilarity(dim=1)
    # Decide device
    gpu_id = get_freer_gpu()
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    options.device = device

def get_batch_sum(data: Any) -> int:
    iter = (i for i in data)
    batch_sum = sum(1 for _ in iter)
    return batch_sum

def main():
    set_global()
    # Create directories to save data
    if options.save:
        # Create directory to save results
        result_dir = resolve_path(options.result_dir, options.identifier)
        os.makedirs(result_dir, exist_ok=True)
        
        # Save options by json format
        save_json(resolve_path(result_dir, "step1.json"), vars(options))

    # Create logger
    if options.save:
        logger = create_logger(f"Step 1", resolve_path(result_dir, "training.log"))
    else:
        logger = create_logger(f"Step 1")

    # Log options
    logger.info(vars(options))

    # Load models
    img_size = get_img_size(options.target_model)

    model = load_model_as_feature_extractor(
        model=options.target_model
    )
    AE = AutoEncoder(get_output_shape(model, img_size))

    if isinstance(model, nn.Module):
        model.to(device) 
        model.eval()
    if isinstance(AE, nn.Module):
        AE.to(device) 
        AE.train()

    _, dataloader = load_fairface_dataset_loader_for_all(options.dataset_dir, \
                                                'train', \
                                                options.data_num, \
                                                options)

    transform= transforms.Compose([
        transforms.Resize((img_size, img_size)),
    ])

    optimizer = optim.Adam(AE.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay)

    batch_sum = get_batch_sum(dataloader)

    loss_function = nn.MSELoss()
    for epoch in range(1, options.n_epochs + 1):
        logger.info(f"[Epoch {epoch:d}/{options.n_epochs:d}]")
        epoch_start_time = time.time()

        for idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            batch = transform_batch_for_extraction(batch[0], transform) 
            batch_feature = model(batch.to(device))
            output = AE(batch_feature)
            loss = loss_function(batch_feature, output)
            logger.info(f"batch {idx} / {batch_sum} loss:{loss}")
            loss.backward()

            optimizer.step()
        epoch_elapsed_time = time.time() - epoch_start_time
        logger.debug(f"[Epcoh {epoch} elapsed time: {epoch_elapsed_time}]")

        # Log elapsed time of the epoch
        epoch_elapsed_time = time.time() - epoch_start_time
        logger.debug(f"[Epcoh {epoch} elapsed time: {epoch_elapsed_time}]")

    if options.save:
        model_path = resolve_path(
            result_dir,
            "AE.pth"
        )
        torch.save(AE.state_dict(), model_path)
        logger.info(f"[Saved model: {model_path}]")

if __name__ == '__main__':
    main()