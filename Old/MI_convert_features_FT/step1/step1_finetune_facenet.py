import os
import sys

sys.path.append("../")
sys.path.append("../../")

import argparse
import copy
import time
import numpy as np
from tqdm import tqdm
import datetime
from logging import Logger

from typing import Any, Tuple, List, Dict

import torch
from torch import optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from utils.celeba import CelebA
from torch import nN
import umap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from utils.util import (save_json,  create_logger, resolve_path, load_model_as_feature_extractor, 
load_model_as_feature_extractor, get_freer_gpu, get_img_size, get_output_shape)

def get_options() -> Any:
    parser = argparse.ArgumentParser()

    # Timestamp
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    parser.add_argument("--identifier", type=str, default=time_stamp, help="timestamp")
    parser.add_argument("--log_interval", type=int, default=100, help="interval between logs (per iteration)")

    # Save
    parser.add_argument("--save", type=bool, default=True, help="if save log and data")

    # GPU
    parser.add_argument("--gpu_idx", type=int, default=0, help="index of cuda devices")

    # Parameteres
    parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
    parser.add_argument("--train", action='store_true', help="training")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate of optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="learning rate of optimizer")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum of optimizer")

    # Directories
    parser.add_argument("--result_dir", type=str, default="~/nas/results/MI_convert_features_FT/step1", help="path to directory which includes results")
    parser.add_argument("--dataset_dir", type=str, default="~/nas/dataset/CelebA_MTCNN160", help="path to directory which includes dataset(Fairface)")

    # Conditions
    parser.add_argument("--embedding_size", type=int, default=128, help="embedding size of features of target model:[128, 512]")
    parser.add_argument("--img_crop", type=int, default=100, help="size of cropping image")
    parser.add_argument("--test_size", type=float, default=0.1, help="ratio of test data to total data")
    parser.add_argument("--val_size", type=float, default=0.1, help="ratio of validation data to total data")
    parser.add_argument("--target_model", type=str, default="FaceNet", 
                        help="target model: 'FaceNet', 'Arcface', 'Magface'")
    parser.add_argument("--target_model_path", type=str, default="/home/akasaka/nas/results/magface_pytorch/2022-05-05-17:57:13_recommended_hyperparams/00001.pth",
    help="path to pth of target model")

    opt = parser.parse_args()

    return opt

def train_target(
    dataloader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    label_map: Dict[int, int],
    logger: Logger,
    log_interval: int
) -> Tuple[float, float]:
    data_num = len(dataloader.dataset)
    loss_avg, correct = 0, 0

    model.train()
    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = torch.tensor([label_map[label.item()] for label in labels]).to(device)

        optimizer.zero_grad()

        out = model(images)
        loss = criterion(out, labels)
        loss_avg += loss
        loss.backward()

        preds = torch.argmax(out, dim=1)
        correct += torch.eq(preds, labels).sum()
        optimizer.step()

        if i % log_interval == 0:
            logger.info(f"[Train] [Loss {loss.item():>7f}] [{i * len(images):5d}/{data_num:5d}]")

    loss_avg /= data_num
    acc_avg = correct / data_num

    logger.info(f"[Train avg] [Loss {loss_avg:>8f} (avg)] [Acc {100 * acc_avg:>0.1f}%, ({correct}/{data_num})]")

    return loss_avg.item(), acc_avg

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

def show_umap(features):
    # UMAP
    embedding = umap.UMAP().fit_transform(features)
    umap_x=embedding[:,0]
    umap_y=embedding[:,1]
    plt.scatter(umap_x, umap_y)
    plt.legend(prop={'size': 5})
    plt.title("UMAP")
    plt.savefig(resolve_path(options.result_dir, 'umap.png'))
    plt.clf()

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

    if options.train:
        model, params_to_update = load_model_as_feature_extractor(
            model=options.target_model,
            embedding_size=options.embedding_size,
            mode='train',
            path=options.target_model_path
        )
    else:
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
    train_data, test_data, val_data, label_map = load_celeba_private_datasets(
        base_dir=options.dataset_dir,
        select=(num_of_identities, num_per_identity),
        test_size=options.test_size, 
        val_size=options.val_size, 
        img_crop_size=options.img_crop,
        img_size=img_size
    )
    train_dataloader = DataLoader(
        train_data,
        batch_size=options.batch_size,
        shuffle=True,
        # Optimization:
        num_workers=os.cpu_count(),
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=options.batch_size,
        shuffle=True,
        # Optimization:
        num_workers=os.cpu_count(),
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_data,
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

    # Create optimizer
    optimizer = optim.SGD(
        params_to_update,
        lr=options.learning_rate,
        momentum=options.momentum,
        weight_decay=options.weight_decay
    )

    # Create loss function
    criterion = nn.CrossEntropyLoss()

    # Train, valid, test model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_acc_test = 0.0
    if options.train:
        for epoch in range(1, options.n_epochs + 1):
            logger.info(f"[Epoch {epoch:d}/{options.n_epochs:d}]")
            epoch_start_time = time.time()

            # Load best model
            model.load_state_dict(best_model_wts)

            # Train model
            train_loss, train_acc = train_target(
                train_dataloader,
                model,
                criterion,
                optimizer,
                label_map,
                logger,
                options.log_interval
            )

            # Validate model
            val_loss, val_acc = eval_target(
                "Valid",
                val_dataloader,
                model,
                criterion,
                label_map,
                logger
            )

            # Test model
            test_loss, test_acc = eval_target(
                "Test",
                test_dataloader,
                model,
                criterion,
                label_map,
                logger
            )

            # Log elapsed time of the epoch
            epoch_elapsed_time = time.time() - epoch_start_time
            logger.debug(f"[Epcoh {epoch} elapsed time: {epoch_elapsed_time}]")

            # Update the best model 
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                logger.info(f"[Update the best model] \
    [Loss(train) {train_loss:>8f}] [Acc(train) {train_acc*100:0.1f}%] \
    [Loss(val) {val_loss:>8f}] [Acc(val) {val_acc*100:0.1f}%] \
    [Loss(test) {test_loss:>8f}] [Acc(test) {test_acc*100:0.1f}%] \
    ")

                if test_acc > best_acc_test:
                    best_acc_test = test_acc

                if options.save:
                    model_path = resolve_path(
                        result_dir,
                        f"{options.target_model}.pth"
                    )
                    torch.save(model.state_dict(), model_path)
                    logger.info(f"[Saved model: {model_path}]")

    if options.train:
        logger.info(f"[Best model: Test acc {test_acc*100:>0.1f}%]")
    
    # Test cosine similarity of two faces of same, and different identity
    model.eval()
    model.classify = False

    output_size = get_output_shape(model, img_size)
    features = np.empty((0, output_size[1]))
    labels = np.empty(0)
    for idx, (data, label) in enumerate(tqdm(test_data)):
        data = data.unsqueeze(0)
        feature = model(data.to(device))
        features = np.append(features, feature.cpu().detach().numpy(), axis=0)
        labels = np.append(labels, [label], axis=-1)


    print(labels)
    
    for label in tqdm(labels):
        plot_features_on_line(features, labels, target_label=label)
    # show_umap(features)
    # save_path = resolve_path(options.result_dir, options.identifier, f'scatter.png')
    # plt.savefig(save_path)
    # print(save_path)

if __name__ == '__main__':
    main()