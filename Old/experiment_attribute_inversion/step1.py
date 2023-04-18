import sys
sys.path.append("../")

import os
import argparse
import numpy as np
import itertools
from tqdm import tqdm 
import datetime
from typing import Any, Tuple, Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

from facenet_pytorch import InceptionResnetV1

from util import save_json, load_json, create_logger, resolve_path
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
    parser.add_argument("--attack_model", type=str, default="ResNet152", help="attack model(feature extraction): 'VGG16', 'ResNet152', 'FaceNet'")
    parser.add_argument("--target_model", type=str, default="FaceNet", help="target model(feature extraction): 'VGG16', 'ResNet152', 'FaceNet'")
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

def extract_feature(batch: Any,
                    device: Any,
                    transform: transforms,
                    model: nn.Module,
                    features: torch.Tensor,
                    layer_name:str):
    batch= transform(batch[0]).to(device)
    feature = model(batch)
    if type(feature) is dict:
        feature = feature[layer_name]
    features = torch.cat((features, feature.detach()), 0)

    return features

def show_umap(features_T: torch.Tensor,
                   features_F: torch.Tensor,
                   attribute_datanum: dict,
                   options: Any):
    # UMAP
    embedding_T = umap.UMAP().fit_transform(features_T.detach().cpu())
    embedding_F = umap.UMAP().fit_transform(features_F.detach().cpu())
    umap_x_T=embedding_T[:,0]
    umap_y_T=embedding_T[:,1]
    umap_x_F=embedding_F[:,0]
    umap_y_F=embedding_F[:,1]
    base_datanum = 0
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False)
    for attribute in attribute_datanum:
        datanum = attribute_datanum[attribute]
        axes[0].scatter((np.mean(umap_x_T[base_datanum:base_datanum+datanum])), np.mean(umap_y_T[base_datanum:base_datanum+datanum]), label=(''.join(attribute) + "_T"))
        axes[1].scatter(np.mean(umap_x_F[base_datanum:base_datanum+datanum]), np.mean(umap_y_F[base_datanum:base_datanum+datanum]), label=(''.join(attribute) + "_F"))
        base_datanum += datanum
    axes[0].legend(prop={'size': 5})
    axes[1].legend(prop={'size': 5})
    plt.title("UMAP")
    plt.savefig(resolve_path(options.result_dir, 'umap.png'))

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

    T, img_size_T, _ = load_model(options.target_model)
    F, img_size_F, layer_name_F = load_model(options.attack_model)
    T.to(device)
    F.to(device)
    T.eval()
    F.eval()

    datasets, dataloaders, attributes_list = load_fairface_dataset_loader(options.dataset_dir, \
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

    attribute_datanum = dict()
    features_T = torch.Tensor().to(device)
    features_F = torch.Tensor().to(device)
    for dataloader_name in tqdm(dataloaders):
        # if 'Black' in dataloader_name or 'White' in dataloader_name or 'East Asian' in dataloader_name:
        for batch in dataloaders[dataloader_name]:
            features_T = extract_feature(batch, device, transform_T, T, features_T, None)
            features_F = extract_feature(batch, device, transform_F, F, features_F, layer_name_F)

        attribute_datanum[dataloader_name] = len(dataloaders[dataloader_name].dataset)

    show_umap(features_T, features_F, attribute_datanum, options)




if __name__ == '__main__':
    main()