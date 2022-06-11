import sys
sys.path.append('../')

import os
import argparse
import numpy as np
import glob
import GPUtil
import time
import itertools
from tqdm import tqdm 
import datetime
from typing import Any, Union, Tuple, Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

from facenet_pytorch import InceptionResnetV1
from sklearn.model_selection import train_test_split

from util import save_json, load_json, create_logger, resolve_path
from fairface import Fairface
import umap
import matplotlib.pyplot as plt

import PIL


def get_options() -> Any:
    parser = argparse.ArgumentParser()

    # Timestamp
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    parser.add_argument("--identifier", type=str, default=time_stamp, help="name of result folder")

    # Folders
    parser.add_argument("--save", type=bool, default=False, help="if save log and data")
    parser.add_argument("--dataset_dir", type=str, default="~/nas/dataset/CelebA_MTCNN160/img_align_celeba", help="path to directory which includes dataset(Fairface)")
    parser.add_argument("--result_dir", type=str, default="~/nas/results/attribute_inversion/step2_celebA", help="path to directory which includes results")
    parser.add_argument("--target_image_dir", type=str, default="~/nas/dataset/target_images", help="path to directory which contains target images")

    # Conditions
    parser.add_argument("--gpu_idx", type=int, default=0, help="index of cuda devices")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--img_channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--data_num", type=int, default=10000, help="number of data to use")
    parser.add_argument("--attack_model", type=str, default="ResNet152", help="attack model(feature extraction): 'VGG16', 'ResNet152', 'FaceNet'")
    parser.add_argument("--target_model", type=str, default="FaceNet", help="target model(feature extraction): 'VGG16', 'ResNet152', 'FaceNet'")
    parser.add_argument("--base_dir_name", type=str, default="1", help="directory name which contains images of target user(features)")
    parser.add_argument("--target_dir_name", type=str, default="1174", help="directory name which contains images of target user(features)")
    parser.add_argument("--attributes", nargs='+', help="attributes to slice the dataset(age, gender, race)")

    opt = parser.parse_args()

    return opt

def load_model(model: str) -> Tuple[nn.Module, int, str]:
    layer_dict = dict()
    img_size_resnet152 = 112
    img_size_facenet = 160

    if model == "ResNet152":
        layer_name = "flatten"
        model = models.resnet152(pretrained=True) # pre-trained on ImageNet
        layer_dict[layer_name] = layer_name
        model = create_feature_extractor(model, layer_dict)
        return model, img_size_resnet152, layer_name
    
    if model == "FaceNet":
        return InceptionResnetV1(classify=False, num_classes=None, pretrained="vggface2"), img_size_facenet, None

    return None

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

def extract_features(imagefolder_path: str,
                     transform: transforms,
                     img_size: int,
                     model: nn.Module,
                     layer_name: str,
                     device: Any):
    all_target_images = torch.tensor([]).to(device)
    all_target_features = torch.tensor([]).to(device)

    for filename in glob.glob(imagefolder_path + "/*.jpg"):
        target_image = PIL.Image.open(filename)
        converted_target_image = transform(target_image).to(device)
        all_target_images= torch.cat((all_target_images, converted_target_image.unsqueeze(0)))

        print(converted_target_image.view(1, -1, img_size, img_size).shape)
        target_feature = model(converted_target_image.view(1, -1, img_size, img_size))
        if type(target_feature) == dict:
            target_feature = target_feature[layer_name]
        target_feature.detach().to(device)
        all_target_features= torch.cat((all_target_features, target_feature.unsqueeze(0)))

    return all_target_features

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
    attribute_group_list = load_json("../attributes.json")

    T, img_size_T, _ = load_model(options.target_model)
    F, img_size_F, layer_name_F = load_model(options.attack_model)
    T.to(device)
    F.to(device)
    T.eval()
    F.eval()

    transform_T = transforms.Compose([
        transforms.Resize((img_size_T, img_size_T)),
        transforms.ToTensor()
    ])
    transform_F = transforms.Compose([
        transforms.Resize((img_size_F, img_size_F)),
        transforms.ToTensor()
    ])
    # Calculate target feature with T (only for features classifier)
    base_imagefolder_path = resolve_path(options.target_image_dir, options.base_dir_name)
    target_imagefolder_path = resolve_path(options.target_image_dir, options.target_dir_name)
    print(target_imagefolder_path)
    all_target_features_T = extract_features(target_imagefolder_path, transform_T, img_size_T, T, None, device)   
    all_target_features_F = extract_features(target_imagefolder_path, transform_F, img_size_F, F, layer_name_F, device)   
    all_base_features_T = extract_features(base_imagefolder_path, transform_T, img_size_T, T, None, device)   
    all_base_features_F = extract_features(base_imagefolder_path, transform_F, img_size_F, F, layer_name_F, device)   

    print(all_target_features_T)
    # print(all_target_features_F)
    # print(all_base_features_T)
    # print(all_base_features_F)


    metric = nn.CosineSimilarity(dim=1)

    base_feature = all_base_features_T[0]
    print("facenet")
    print("base")
    for idx, feature in enumerate(all_base_features_T):
        # print(feature.shape)
        if idx == 0:
            continue
        cos_sim = metric(base_feature, feature)
        print(cos_sim)

    print("target")
    for idx, feature in enumerate(all_target_features_T):
        if idx == 0:
            continue
        cos_sim = metric(base_feature, feature)
        print(cos_sim)

    base_feature = all_base_features_F[0]
    print("resnet")
    print("base")
    for idx, feature in enumerate(all_base_features_F):
        if idx == 0:
            continue
        cos_sim = metric(base_feature, feature)
        print(cos_sim)

    print("target")
    for idx, feature in enumerate(all_target_features_F):
        if idx == 0:
            continue
        cos_sim = metric(base_feature, feature)
        print(cos_sim)


if __name__ == '__main__':
    main()
