import sys

sys.path.append("../")

import os
import itertools
import argparse
import glob
import time
import datetime
import PIL
from tqdm import tqdm 
from collections import Counter
from typing import Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch import linalg as LA
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from fairface import Fairface

from util import (save_json, load_json, create_logger, resolve_path, load_model_as_feature_extractor, RandomBatchLoader,
    load_attacker_discriminator, load_attacker_generator, get_freer_gpu, get_img_size, extract_features_from_nnModule)

from celeba import CelebA


def get_options() -> Any:
    parser = argparse.ArgumentParser()

    # Timestamp
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    parser.add_argument("--identifier", type=str, default=time_stamp, help="timestamp")

    # Save
    parser.add_argument("--save", type=bool, default=True, help="if save log and data")

    # Directories
    parser.add_argument("--result_dir", type=str, default="~/nas/results/MI_attribute_average/step3", help="path to directory which includes results")
    parser.add_argument("--step2_dir", type=str, default="pure_facenet_500epoch_features", help="path to directory which includes the step2 result")
    parser.add_argument("--target_image_dir", type=str, default="~/nas/dataset/target_images", help="path to directory which contains target images")
    parser.add_argument("--dataset_dir", type=str, default="~/nas/dataset/fairface", help="path to directory which includes dataset(Fairface)")

    # For inference
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--times", type=int, default=5, help="times to initialize z")
    parser.add_argument("--epochs", type=int, default=1500, help="times to initialize z") 
    parser.add_argument("--iterations", type=int, default=8, help="iterations to optimize z")
    parser.add_argument("--learning_rate", type=float, default=0.02, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="learning rate")
    parser.add_argument("--lambda_i", type=float, default=150, help="learning rate")

    # Conditions
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--data_num", type=int, default=10000, help="number of data to use")
    parser.add_argument("--target_model", type=str, default="FaceNet", help="target model: 'FaceNet', 'Arcface'")
    parser.add_argument("--attack_model", type=str, default="Arcface", help="attack model: 'FaceNet', 'Arcface'")
    parser.add_argument("--single_mode", type=bool, default=True, help="True if you use just one stored feature(features)")
    parser.add_argument("--attributes", nargs='+', default="gender", help="attributes to slice the dataset(age, gender, race)")

    opt = parser.parse_args()

    return opt

def L_prior(D: nn.Module, G: nn.Module, z: torch.Tensor) -> torch.Tensor:
    return torch.mean(-D(G(z)))


def calc_id_loss(G: nn.Module, FE: nn.Module, z: torch.Tensor, device: str, all_target_features: torch.Tensor, image_size: int) -> torch.Tensor:
    resize = transforms.Resize((image_size, image_size))
    metric = nn.CosineSimilarity(dim=1)
    Gz_features = FE(resize(G(z))).to(device)
    dim = Gz_features.shape[1]

    sum_of_cosine_similarity = 0
    for target_feature in all_target_features.view(-1, dim):
        target_feature = target_feature.expand(Gz_features.shape[0], -1)
        sum_of_cosine_similarity += metric(target_feature, Gz_features)
    
    return 1 - torch.mean(sum_of_cosine_similarity / all_target_features.shape[0])

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

def transform_batch_for_extraction(batch: list,
                                   transform: transforms) -> torch.Tensor:
    return transform(batch)

options = get_options()

def main():

    # Decide device
    gpu_id = get_freer_gpu()
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    options.device = device

    # Load step1 and step2 options
    step2_dir = "~/nas/results/common/step2/" + options.step2_dir
    step1_options = load_json(resolve_path(step2_dir, "step1.json"))
    step2_options = load_json(resolve_path(step2_dir, "step2.json"))
    label_map = load_json(resolve_path(step2_dir, "label_map.json"))

    # Create directories to save data
    if options.save:
        # Create directory to save results
        result_dir = resolve_path(options.result_dir, options.identifier)
        os.makedirs(result_dir, exist_ok=True)
        
        # Create directory to save each result
        for i in range(1, options.times + 1):
            result_each_time_dir = resolve_path(result_dir, f"{i}")
            os.makedirs(result_each_time_dir, exist_ok=True)

        # Save step1 and step2 options
        save_json(resolve_path(result_dir, "step1.json"), step1_options)
        save_json(resolve_path(result_dir, "step2.json"), step2_options)
        save_json(resolve_path(result_dir, "label_map.json"), label_map)

        # Save options by json format
        save_json(resolve_path(result_dir, "step3.json"), vars(options))

    # Create logger
    if options.save:
        logger = create_logger(f"Step 3", resolve_path(result_dir, "training.log"))
    else:
        logger = create_logger(f"Step 3")

    # Log options
    logger.info(vars(options))

    # Load attribute list
    attribute_group_list = load_json("attributes.json")

    # Load models
    img_size_T = get_img_size(options.target_model)
    img_size_A = get_img_size(options.attack_model)
    img_shape_T = (options.img_channels, img_size_T, img_size_T)
    img_shape_A = (options.img_channels, img_size_A, img_size_A)

    D_T = load_attacker_discriminator(
        path=resolve_path(step2_dir, "D.pth"),
        input_dim=step2_options["img_channels"],
        network_dim=step2_options["D_network_dim"],
        img_shape=img_shape_T
    )
    D_A = load_attacker_discriminator(
        path=resolve_path(step2_dir, "D.pth"),
        input_dim=step2_options["img_channels"],
        network_dim=step2_options["D_network_dim"],
        img_shape=img_shape_A
    )
    G_T = load_attacker_generator(
        path=resolve_path(step2_dir, "G.pth"),
        latent_dim=options.latent_dim,
        network_dim=step2_options["G_network_dim"],
        img_shape=img_shape_T
    )
    G_A = load_attacker_generator(
        path=resolve_path(step2_dir, "G.pth"),
        latent_dim=options.latent_dim,
        network_dim=step2_options["G_network_dim"],
        img_shape=img_shape_A
    )
    T = load_model_as_feature_extractor(
        model=options.target_model
    )
    A = load_model_as_feature_extractor(
        model=options.attack_model
    )
    
    if isinstance(D_T, nn.Module):
        D_T.to(device) 
        D_T.eval()
    if isinstance(D_A, nn.Module):
        D_A.to(device) 
        D_A.eval()
    if isinstance(G_T, nn.Module):
        G_T.to(device) 
        G_T.eval()
    if isinstance(G_A, nn.Module):
        G_A.to(device) 
        G_A.eval()
    if isinstance(T, nn.Module):
        T.to(device) 
        T.eval()
    if isinstance(A, nn.Module):
        A.to(device) 
        A.eval()

    # Load datasets
    _, dataloaders, _= load_fairface_dataset_loader(options.dataset_dir, \
                                                'train', \
                                                options.data_num, \
                                                attribute_group_list, \
                                                options)

    transform_T = transforms.Compose([
        transforms.Resize((img_size_T, img_size_T)),
    ])
    transform_A = transforms.Compose([
        transforms.Resize((img_size_A, img_size_A)),
    ])

    # Extract features from model
    attribute_to_feature_T = dict()
    attribute_to_feature_A = dict()
    for dataloader_name in tqdm(dataloaders):
        features_T = torch.Tensor().to(device)
        features_A = torch.Tensor().to(device)
        for batch in dataloaders[dataloader_name]:
            batch_T = transform_batch_for_extraction(batch[0], transform_T) 
            batch_A = transform_batch_for_extraction(batch[0], transform_A) 
            feature_T = extract_features_from_nnModule(batch_T,  T, None, device)
            feature_A = extract_features_from_nnModule(batch_A,  A, None, device)
            features_T = torch.cat((features_T, feature_T.detach()), 0)
            features_A = torch.cat((features_A, feature_A.detach()), 0)
        attribute_to_feature_T[dataloader_name] = features_T
        attribute_to_feature_A[dataloader_name] = features_A


    # Calculate averages of features
    # attribute_to_distance_dict[attribute][attribute] = distance (distance from all features of first attribute to the average feature of second attribute)
    attribute_to_average_T = dict()
    attribute_to_average_A = dict()
    for attribute in attribute_to_feature_T: # or A
        average_of_features_T = calculate_average_of_features(attribute_to_feature_T, attribute)
        average_of_features_A = calculate_average_of_features(attribute_to_feature_A, attribute)
        attribute_to_average_T[attribute] = average_of_features_T
        attribute_to_average_A[attribute] = average_of_features_A

    attribute = list(attribute_to_feature_T.keys())[0]
    target_feature_T = attribute_to_average_T[attribute]
    target_feature_A = attribute_to_average_A[attribute]

    target_feature_set = [("T", T, G_T, D_T, target_feature_T, img_size_T),\
                          ("A", A, G_A, D_A, target_feature_A, img_size_A)]

    for name, model, G, D, target_feature, img_size in target_feature_set:
        # Search z^
        for i in range(1, options.times + 1):
            z = torch.randn(options.batch_size * options.iterations, options.latent_dim, requires_grad=True, device=device)
            optimizer = optim.SGD([z], lr=options.learning_rate, momentum=options.momentum)
            dataloader = RandomBatchLoader(z, options.batch_size)

            # Optimize z
            start_time = time.time()
            for epoch in range(1, options.epochs + 1):
                L_prior_loss_avg, L_id_loss_avg, total_loss_avg = 0, 0, 0

                for batch_idx, batch in enumerate(dataloader):
                    optimizer.zero_grad()

                    L_prior_loss = L_prior(D, G, batch)
                    L_id_loss = calc_id_loss(G, model, batch, device, target_feature, img_size) 
                    
                    L_id_loss = options.lambda_i * L_id_loss
                    total_loss = L_prior_loss + L_id_loss

                    L_prior_loss_avg += L_prior_loss
                    L_id_loss_avg += L_id_loss
                    total_loss_avg += total_loss

                    total_loss.backward()
                    optimizer.step()

                L_prior_loss_avg /= z.shape[0]
                L_id_loss_avg /= z.shape[0]
                total_loss_avg /= z.shape[0]

                logger.info(f"[{i}/{options.times}], \
                            [{epoch}/{options.epochs}],\
                            [Loss: {total_loss_avg.item()},\
                            L_prior: {L_prior_loss_avg.item()},\
                            L_id: {L_id_loss_avg.item()}]")

            elapsed_time = time.time() - start_time
            logger.debug(f"[Elapsed time of all epochs: {elapsed_time}]")
                
            # Save results
            if options.save:
                result_dataloader = RandomBatchLoader(z, options.batch_size)
                result_each_time_dir = resolve_path(result_dir, f"{i}")

                all_best_images = torch.tensor([]).to(device)

                # Calc 
                for _, batch in enumerate(result_dataloader):
                    images = G(batch)

                    best_image, _ = get_best_image(model, images, img_size, target_feature)

                    all_best_images = torch.cat((all_best_images, best_image.unsqueeze(0)))


                # Save all best images
                best_images_path = resolve_path(result_each_time_dir, f"best_images_{name}_{''.join(attribute)}.png")
                save_image(all_best_images, best_images_path, normalize=True, nrow=options.iterations)
                logger.info(f"[Saved all best images: {best_images_path}]")

if __name__ == '__main__':
	main()