import os
import sys

sys.path.append("../")
sys.path.append("../../")

import argparse
import numpy as np
from tqdm import tqdm
import datetime
from logging import Logger

from typing import Any, Tuple, List, Dict

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from util.celeba import CelebA
from torch import nn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from util.util import (save_json, create_logger, resolve_path, load_model_as_feature_extractor, 
load_model_as_feature_extractor, get_freer_gpu, get_img_size, get_output_shape)

def get_options() -> Any:
    parser = argparse.ArgumentParser()

    # Timestamp
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    parser.add_argument("--identifier", type=str, default=time_stamp, help="timestamp")

    # Save
    parser.add_argument("--save", type=bool, default=True, help="if save log and data")

    # GPU
    parser.add_argument("--gpu_idx", type=int, default=0, help="index of cuda devices")

    # Parameteres
    parser.add_argument("--train", action='store_true', help="training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")

    # Directories
    parser.add_argument("--result_dir", type=str, default="~/nas/results/MI_convert_features_FT/step1", help="path to directory which includes results")
    parser.add_argument("--dataset_dir", type=str, default="~/nas/dataset/CelebA_MTCNN160", help="path to directory which includes dataset(Fairface)")

    # Conditions
    parser.add_argument("--embedding_size", type=int, default="128", help="embedding size of features of target model:[128, 512]")
    parser.add_argument("--img_crop", type=int, default=100, help="size of cropping image")
    parser.add_argument("--test_size", type=float, default=0.1, help="ratio of test data to total data")
    parser.add_argument("--val_size", type=float, default=0.1, help="ratio of validation data to total data")
    parser.add_argument("--target_model", type=str, default="FaceNet", help="target model: 'FaceNet', 'Arcface', 'Magface'")
    parser.add_argument("--target_model_path", type=str, help="path to pth of target model")

    opt = parser.parse_args()

    return opt

def eval_target(
    mode: str,
    dataloader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    label_map: Dict[int, int],
    logger: Logger
) -> Tuple[float, float]:
    data_num = len(dataloader.dataset)
    loss_avg, correct = 0, 0

    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = torch.tensor([label_map[label.item()] for label in labels]).to(device)

            outputs = model(images)
            loss_avg += criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1, keepdim=True)
            correct += torch.eq(preds, labels.view_as(preds)).sum().item()

    loss_avg /= data_num
    acc_avg = correct / data_num

    logger.info(f"[{mode} avg] [Loss {loss_avg:>8f} (avg)] [Acc {100 * acc_avg:>0.1f}%, ({correct}/{data_num})]")

    return loss_avg.item(), acc_avg

def compress_labels(labels: List[int]) -> Dict[int, int]:
    label_map = {}

    index = 0
    for label in labels:
        if label not in label_map:
            label_map[label] = index
            index += 1

    return label_map

def load_celeba_private_datasets(
    base_dir: str,
    select: Tuple[int, int],
    test_size: float,
    val_size: float,
    img_crop_size: int,
    img_size: int
) -> Tuple[CelebA, CelebA, CelebA, Dict[int, int]]:
    transform = transforms.Compose([
        # transforms.CenterCrop(img_crop_size),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    dataset = CelebA(
        base_dir=base_dir,
        usage='train',
        select=select,
        transform=transform
    )

    label_map = compress_labels([label.item() for label in dataset.labels])

    training_indices, test_indices = train_test_split(
        list(range(len(dataset))),
        test_size=test_size,
        stratify=dataset.labels,
        random_state=0
    )

    training_indices, val_indices = train_test_split(
        training_indices, 
        test_size=val_size/(1-test_size),
        stratify=dataset.labels[training_indices],
        random_state=0
    )

    training_dataset = Subset(dataset, training_indices)
    test_dataset = Subset(dataset, test_indices)
    val_dataset = Subset(dataset, val_indices)

    return training_dataset, test_dataset, val_dataset, label_map

def plot_features_on_line(features, labels, target_label):
    features_from_test_data = lambda: zip(torch.from_numpy(features), labels)
    metrics = nn.CosineSimilarity(dim=0)
    # Get image
    sames = []
    diffs = []
    for feature, label in features_from_test_data():
        if target_label == label:
            target_feature = feature
        
    for feature, label in features_from_test_data():
        if target_label == label:
            dist = metrics(target_feature, feature)
            # print("Cos sim for same face: ", dist)
            sames.append(dist.cpu().detach().numpy())
        else:
            dist = metrics(target_feature, feature)
            # print("Cos sim for dif face: ", dist)
            diffs.append(dist.cpu().detach().numpy())
        
    plt.plot(sames, np.full(len(sames), 1), 'o')
    plt.plot(diffs, np.full(len(diffs), 0), 'x')
    save_path = resolve_path(options.result_dir, options.identifier, f'scatter_{target_label}.png')
    plt.savefig(save_path)
    plt.clf()
    # print(save_path)

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
    num_of_identities = 1000
    num_per_identity = 30

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

    model, params_to_update = load_model_as_feature_extractor(
        model=options.target_model,
        embedding_size=options.embedding_size,
        mode='eval',
        path=options.target_model_path
    )

    if isinstance(model, nn.Module):
        model.to(device) 
        model.train()

    # Load private datasets
    _, test_data, _, label_map = load_celeba_private_datasets(
        base_dir=options.dataset_dir,
        select=(num_of_identities, num_per_identity),
        test_size=options.test_size, 
        val_size=options.val_size, 
        img_crop_size=options.img_crop,
        img_size=img_size
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=options.batch_size,
        shuffle=True,
        # Optimization:
        num_workers=os.cpu_count(),
        pin_memory=True
    )

    # Save label mappings
    if options.save:
        # Save label mappings by json format
        save_json(resolve_path(result_dir, "label_map.json"), label_map)

    # Create loss function
    criterion = nn.CrossEntropyLoss()

    # Train, valid, test model
    test_loss, test_acc = eval_target(
        "Test",
        test_dataloader,
        model,
        criterion,
        label_map,
        logger
    )
    logger.info(f"[Loss(test) {test_loss:>8f}] [Acc(test) {test_acc*100:0.1f}%]")


    # Test cosine similarity of two faces of same, and different identity
    model.eval()
    model.classify = False

    output_size = get_output_shape(model, img_size)
    features = np.empty((0, output_size[1]))
    labels = np.empty(0)
    for _, (data, label) in enumerate(tqdm(test_data)):
        data = data.unsqueeze(0)
        feature = model(data.to(device))
        features = np.append(features, feature.cpu().detach().numpy(), axis=0)
        labels = np.append(labels, [label], axis=-1)
    
    for label in tqdm(labels):
        plot_features_on_line(features, labels, target_label=label)

if __name__ == '__main__':
    main()