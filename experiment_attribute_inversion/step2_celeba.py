import sys

sys.path.append("../")

import os
import argparse
import datetime
from typing import Any, Union, Tuple, Dict, List

import torch
from torch import nn
import torchvision.transforms as transforms


from util import save_json, load_model_as_feature_extraction, extract_features_from_nnModule, extract_features_from_orx,create_batch_from_images, create_logger, resolve_path

def get_options() -> Any:
    parser = argparse.ArgumentParser()

    # Timestamp
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    parser.add_argument("--identifier", type=str, default=time_stamp, help="name of result folder")

    # Folders
    parser.add_argument("--save", type=bool, default=False, help="if save log and data")
    parser.add_argument("--dataset_dir", type=str, default="~/nas/dataset/CelebA_MTCNN160/img_align_celeba", \
        help="path to directory which includes dataset(Fairface)")
    parser.add_argument("--result_dir", type=str, default="~/nas/results/attribute_inversion/step2_celebA", \
        help="path to directory which includes results") 
    parser.add_argument("--target_image_dir", type=str, default="~/nas/dataset/target_images", \
        help="path to directory which contains target images")

    # Conditions
    parser.add_argument("--gpu_idx", type=int, default=0, help="index of cuda devices")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--img_channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--data_num", type=int, default=10000, help="number of data to use")
    parser.add_argument("--attack_model", type=str, default="Arcface", help="attack model: \
        'Arcface', 'FaceNet'")
    parser.add_argument("--target_model", type=str, default="FaceNet", help="target model: \
        'Arcface', 'FaceNet'")
    parser.add_argument("--base_dir_name", type=str, default="2000", \
        help="directory name which contains images of target user(features)")
    parser.add_argument("--target_dir_name", type=str, default="1174", \
        help="directory name which contains images of target user(features)")
    parser.add_argument("--attributes", nargs='+', help="attributes to slice the dataset(age, gender, race)")

    opt = parser.parse_args()

    return opt


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

def calculate_cos_sim(base_feature: torch.Tensor,
                  features: torch.Tensor):
    metric = nn.CosineSimilarity()
    cos_sim = metric(base_feature.unsqueeze(0), features)
    return cos_sim


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

    T, img_size_T, _ = load_model_as_feature_extraction(options.target_model, options)
    F, img_size_F, _ = load_model_as_feature_extraction(options.attack_model, options)

    if isinstance(T, nn.Module):
        T.to(device) 
        T.eval()
    if isinstance(F, nn.Module):
        F.to(device) 
        F.eval()

    transform_T = transforms.Compose([
        transforms.Resize((img_size_T, img_size_T)),
        transforms.ToTensor()
    ])
    transform_F = transforms.Compose([
        transforms.Resize((img_size_F, img_size_F)),
        transforms.ToTensor()
    ])

    # Extract features
    base_imagefolder_path = resolve_path(options.target_image_dir, options.base_dir_name)
    target_imagefolder_path = resolve_path(options.target_image_dir, options.target_dir_name)

    target_batch_T = create_batch_from_images(target_imagefolder_path, transform_T)
    base_batch_T = create_batch_from_images(base_imagefolder_path, transform_T)
    target_batch_F = create_batch_from_images(target_imagefolder_path, transform_F)
    base_batch_F = create_batch_from_images(base_imagefolder_path, transform_F)

    target_features_T = extract_features_from_nnModule(target_batch_T, T, None, device)
    base_features_T = extract_features_from_nnModule(base_batch_T, T, None, device)
    target_features_F = extract_features_from_nnModule(target_batch_F, F, None, device)
    base_features_F = extract_features_from_nnModule(base_batch_F, F, None, device)

    print(base_features_F)
    print(target_features_F)


    # Calculate cosine similarity
    base_feature = base_features_T[0]
    print(options.target_model)
    print("base")
    print(torch.mean(calculate_cos_sim(base_feature, base_features_T)))
    print("target")
    print(torch.mean(calculate_cos_sim(base_feature, target_features_T)))
    base_feature = base_features_F[0]
    print(options.attack_model)
    print("base")
    print(torch.mean(calculate_cos_sim(base_feature, base_features_F)))
    print("target")
    print(torch.mean(calculate_cos_sim(base_feature, target_features_F)))


if __name__ == '__main__':
    main()
