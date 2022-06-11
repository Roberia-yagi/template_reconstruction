# Train converter
import sys

from sklearn import datasets


sys.path.append("../")
sys.path.append("../../")

import os
import itertools
import argparse
import copy
import numpy as np
from random import randint
from termcolor import cprint
import matplotlib.pyplot as plt
import time
import datetime
from logging import Logger
from tqdm import tqdm 
from typing import Any, List, Tuple, Dict
from torch.utils.data.dataset import Subset

from utils.email_sender import send_email
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from utils.celeba import CelebA 
from utils.casia_web_face import CasiaWebFace

from utils.util import (save_json, load_json, create_logger, resolve_path, load_model_as_feature_extractor, BatchLoader,
    get_freer_gpu, get_img_size, extract_features_from_nnModule, load_autoencoder)

def get_options() -> Any:
    parser = argparse.ArgumentParser()

    # Timestamp
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    parser.add_argument("--identifier", type=str, default=time_stamp, help="timestamp")
    parser.add_argument("--gpu_idx", type=int, default=0, help="index of cuda devices")

    # Save
    parser.add_argument("--log_interval", type=int, default=100, help="interval between logs (per iteration)")

    # Directories
    parser.add_argument("--result_dir", type=str, default="../../../results/train_transferer_cossim", help="path to directory which includes results")
    parser.add_argument("--dataset_dir", type=str, default="../../../dataset/CASIAWebFace_MTCNN160", help="path to directory which includes dataset(Fairface)")
    parser.add_argument('--target_model_path', type=str, help='path to pretrained target model')
    parser.add_argument('--attack_model_path', type=str, help='path to pretrained attack model')
    parser.add_argument('--AE_path', type=str, help='path to pretrained AE')

    # Conditions of optimizer
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate of optimizer")
    parser.add_argument("--beta1", type=float, default=0.9, help="hyper-parameter of Adam")
    parser.add_argument("--beta2", type=float, default=0.999, help="hyper-parameter of Adam")
    parser.add_argument("--weight_decay", type=float, default=0, help="hyper-parameter of Adam")


    # Conditions of Model
    parser.add_argument("--embedding_size", type=int, default=512, help="embedding size of features of target model:[128, 512]")
    parser.add_argument("--target_model", type=str, default="Magface", help="target model: 'FaceNet', 'Arcface', 'Magface")
    parser.add_argument("--attack_model", type=str, default="FaceNet", help="attack model: 'FaceNet', 'Arcface', 'Magface")
    parser.add_argument("--AE_ver", type=int, default=1.1, help="AE version: 1, 1.1, 1.2, 1.3, 1.4")

    # Conditions of training
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--n_epochs", type=int, default=int(1e9), help="number of epochs of training")
    parser.add_argument("--resume", action='store_true', help='flag of resume')
    parser.add_argument("--resume_epoch", type=int, help='flag of resume')
    parser.add_argument("--num_of_identities", type=int, default=7000, help="Number of unique identities")
    parser.add_argument("--num_per_identity", type=int, default=20, help="Number of unique identities")

    opt = parser.parse_args()

    return opt

def compress_labels(labels: List[int]) -> Dict[int, int]:
    label_map = {}

    index = 0
    for label in labels:
        if label not in label_map:
            label_map[label] = index
            index += 1

    return label_map


def train_target(
    dataloader: DataLoader,
    T: nn.Module,
    A: nn.Module,
    transform_T: transforms,
    transform_A: transforms,
    AE: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    logger: Logger,
    log_interval: int
) -> Tuple[float, float]:
    data_num = len(dataloader.dataset)
    loss_avg = 0

    AE.train()
    for i, (images, _) in enumerate(dataloader):
        images = images.to(device)
        # images = images.double().to(device)

        optimizer.zero_grad()

        features_T = T(transform_T(images))
        converted_features_in_A = AE(features_T)
        features_A = A(transform_A(images))
        loss = (1 - criterion(features_A, converted_features_in_A).mean()) / 2
        # DELETE
        if i == 0:
            cprint(converted_features_in_A, 'green')
            cprint(features_A, 'red')
        loss_avg += loss
        loss.backward()

        optimizer.step()

        if i % log_interval == 0:
            logger.info(f"[Train] [Loss {loss.item():.8f}] [{i * len(images):5d}/{data_num:5d}]")

    loss_avg /= i

    logger.info(f"[Train avg] [Loss {loss_avg:.10f} (avg)]")

    return loss_avg.item()


def eval_target(
    mode: str,
    dataloader: DataLoader,
    T: nn.Module,
    A: nn.Module,
    transform_T: transforms,
    transform_A: transforms,
    AE: nn.Module,
    criterion: nn.Module,
    logger: Logger
) -> Tuple[float, float]:
    data_num = len(dataloader.dataset)
    loss_avg = 0

    AE.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            # images = images.double().to(device)

            features_T = T(transform_T(images))
            converted_features_in_A = AE(features_T)
            features_A = A(transform_A(images))
            loss = (1 - criterion(features_A, converted_features_in_A).mean()) / 2
            loss_avg += loss

    loss_avg /= i

    logger.info(f"[{mode} avg] [Loss {loss_avg:.10f} (avg)]")

    return loss_avg.item()


def set_global():
    global options
    global device

    options = get_options()
    # Decide device
    # gpu_id = get_freer_gpu()
    # device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    # Decide device
    device = f"cuda:{options.gpu_idx}" if torch.cuda.is_available() else "cpu"

    options.device = device

def main():

    set_global()

    if options.resume:
        if options.resume_epoch is None:
            print('Specify resume_epoch')
            exit(1)

    # Create directory to save results
    result_dir = resolve_path(options.result_dir, (options.identifier + f'_{options.num_of_identities}_identities'))
    if os.path.exists(result_dir):
        result_dir = result_dir + randint(0, 1e9)
        os.makedirs(result_dir, exist_ok=False)
    else:
        os.makedirs(result_dir, exist_ok=False)
    
    # Save options by json format
    save_json(resolve_path(result_dir, "step1.json"), vars(options))

    # Create logger
    logger = create_logger(f"Step 1", resolve_path(result_dir, "training.log"))

    # Log options
    logger.info(vars(options))

    # Load models
    img_size_T = get_img_size(options.target_model)
    img_size_A = get_img_size(options.attack_model)

    AE = load_autoencoder(
        pretrained=options.resume,
        model_path=options.AE_path,
        mode='train',
        ver=options.AE_ver
    )
    T, _ = load_model_as_feature_extractor(
        arch=options.target_model,
        embedding_size=options.embedding_size,
        mode='eval',
        path=options.target_model_path,
        pretrained=True
    )
    A, _ = load_model_as_feature_extractor(
        arch=options.attack_model,
        embedding_size=options.embedding_size,
        mode='eval',
        path=options.attack_model_path,
        pretrained=True
    )

    logger.info(f'Architecture of AE is {AE}')

    if isinstance(AE, nn.Module):
        AE.to(device) 
        AE.train()
    if isinstance(T, nn.Module):
        T.to(device) 
    if isinstance(A, nn.Module):
        A.to(device) 

    train_dataset = CasiaWebFace(base_dir=options.dataset_dir,
                           usage='train',
                           num_of_identities=options.num_of_identities,
                           num_per_identity=options.num_per_identity,
                           transform=transforms.ToTensor())
    test_dataset = CasiaWebFace(base_dir=options.dataset_dir,
                           usage='test',
                           num_of_identities=options.num_of_identities,
                           num_per_identity=options.num_per_identity,
                           transform=transforms.ToTensor())
    val_dataset = CasiaWebFace(base_dir=options.dataset_dir,
                           usage='valid',
                           num_of_identities=options.num_of_identities,
                           num_per_identity=options.num_per_identity,
                           transform=transforms.ToTensor())

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=options.batch_size,
        shuffle=True,
        # Optimization:
        num_workers=os.cpu_count(),
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=options.batch_size,
        shuffle=True,
        # Optimization:
        num_workers=os.cpu_count(),
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=options.batch_size,
        shuffle=False,
        # Optimization:
        num_workers=os.cpu_count(),
        pin_memory=True
    )

    transform_T = transforms.Compose([
        transforms.Resize((img_size_T, img_size_T)),
    ])
    transform_A = transforms.Compose([
        transforms.Resize((img_size_A, img_size_A)),
    ])

    optimizer = optim.Adam(AE.parameters(), lr=options.learning_rate, betas=(options.beta1, options.beta2), weight_decay=options.weight_decay)
    criterion = nn.CosineSimilarity()


    # Train, valid, test model
    best_model_wts = copy.deepcopy(AE.state_dict())
    best_loss = 1e9
    best_loss_test = 1e9
    loss_update_counter = 0
    if options.resume:
        start_epoch = options.resume_epoch
    else:
        start_epoch = 0
    for epoch in range(start_epoch + 1, options.n_epochs + 1):
        if loss_update_counter > 10:
            break
        logger.info(f"[Epoch {epoch:d}/{options.n_epochs:d}]")
        epoch_start_time = time.time()

        # Load best model
        AE.load_state_dict(best_model_wts)

        # Train model
        train_loss= train_target(
            train_dataloader,
            T,
            A,
            transform_T,
            transform_A,
            AE,
            criterion,
            optimizer,
            logger,
            options.log_interval
        )

        # Validate model
        val_loss= eval_target(
            "Valid",
            val_dataloader,
            T,
            A,
            transform_T,
            transform_A,
            AE,
            criterion,
            logger
        )

        # Test model
        test_loss= eval_target(
            "Test",
            test_dataloader,
            T,
            A,
            transform_T,
            transform_A,
            AE,
            criterion,
            logger
        )

        # Log elapsed time of the epoch
        epoch_elapsed_time = time.time() - epoch_start_time
        logger.debug(f"[Epcoh {epoch} elapsed time: {epoch_elapsed_time}]")

        # Update the best model 
        if val_loss < best_loss:
            loss_update_counter = 0
            best_loss = val_loss
            best_model_wts = copy.deepcopy(AE.state_dict())
            logger.info(f"[Update the best model] \
        [Loss(train) {train_loss:.10f}] \
        [Loss(val) {val_loss:.10f}] \
        [Loss(test) {test_loss:.10f}] \
        ")
            if test_loss < best_loss_test:
                best_loss_test = test_loss

            model_path = resolve_path(
                result_dir,

                "AE.pth"
            )
            torch.save(AE.state_dict(), model_path)
            logger.info(f"[Saved model: {model_path}]")
        else:
            loss_update_counter += 1

        if epoch % 10 == 0:
            torch.save(AE.state_dict(), resolve_path(result_dir, f"AE_{epoch}.pth"))
    
if __name__ == '__main__':
    main()