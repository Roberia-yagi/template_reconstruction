# Train converter
import sys

sys.path.append("../")
sys.path.append("../../")

import os
import argparse
import copy
import numpy as np
from random import randint
import time
import datetime
from logging import Logger
from typing import Any, Tuple

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.casia_web_face import CasiaWebFace

from utils.util import (save_json, create_logger, resolve_path, load_model_as_feature_extractor,
    get_img_size, load_autoencoder, get_freer_gpu)

def get_options() -> Any:
    parser = argparse.ArgumentParser()

    # Timestamp
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    parser.add_argument("--identifier", type=str, default=time_stamp, help="timestamp")
    parser.add_argument("--gpu_idx", type=int, default=1, help="index of cuda devices")

    # Save
    parser.add_argument("--log_interval", type=int, default=100, help="interval between logs (per iteration)")

    # Directories
    parser.add_argument("--result_dir", type=str, default="../../../results/train_transferer_cossim", help="path to directory which includes results")
    parser.add_argument("--dataset_dir", type=str, default="../../../dataset/CASIAWebFace_MTCNN160", help="path to directory which includes dataset(Fairface)")
    parser.add_argument('--target_model_path', type=str, help='path to pretrained target model')
    parser.add_argument('--attack_model_path', type=str, help='path to pretrained attack model')
    parser.add_argument('--AE_path', type=str, help='path to pretrained AE')

    # Conditions of optimizer
    parser.add_argument("--learning_rate", "--lr", type=float, default=0.001, help="learning rate of optimizer")
    parser.add_argument("--beta1", type=float, default=0.9, help="hyper-parameter of Adam")
    parser.add_argument("--beta2", type=float, default=0.999, help="hyper-parameter of Adam")
    parser.add_argument("--weight_decay", type=float, default=0, help="hyper-parameter of Adam")


    # Conditions of Model
    parser.add_argument("--target_embedding_size", type=int, default=512, help="embedding size of features of target model:[128, 512]")
    parser.add_argument("--attack_embedding_size", type=int, default=512, help="embedding size of features of target model:[128, 512]")
    parser.add_argument("--target_model", type=str, default="FaceNet", help="target model: 'FaceNet', 'Arcface', 'Magface")
    parser.add_argument("--attack_model", type=str, default="Magface", help="attack model: 'FaceNet', 'Arcface', 'Magface")
    parser.add_argument("--AE_ver", type=float, default=1.1, help="AE version: 1, 1.1, 1.2, 1.3, 1.4")

    # Conditions of training
    parser.add_argument("--n_epochs", type=int, default=int(1e9), help="number of epochs of training")
    parser.add_argument("--resume", action='store_true', help='flag of resume')
    parser.add_argument("--resume_epoch", type=int, help='flag of resume')
    parser.add_argument("--num_of_identities", type=int, default=7000, help="Number of unique identities")
    parser.add_argument("--num_per_identity", type=int, default=20, help="Number of unique identities")
    parser.add_argument("--early_stop", type=int, default=1, help="the trial limitation of non-update training")
    parser.add_argument("--iteration", type=int, default=5, help="the number of training for fair evaluation")

    opt = parser.parse_args()

    return opt


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
    loss_history = np.array([])

    AE.train()
    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)

        optimizer.zero_grad()

        feature_idx = randint(0, options.num_per_identity - 1)

        # Extract features and convert features in the target model space to in the attack model space
        target_features_T = T(transform_T(images))
        converted_target_features_in_A = AE(target_features_T)
        target_features_in_A = A(transform_A(images))
        same_loss = torch.abs((criterion(target_features_in_A[feature_idx], converted_target_features_in_A[0].unsqueeze(0)) -
            criterion(target_features_in_A[feature_idx], target_features_in_A[0].unsqueeze(0)))).mean() / 2

        # Sato
        # same_loss = (1 - criterion((target_features_in_A[feature_idx] - converted_target_features_in_A),
        #     (target_features_in_A[feature_idx] - target_features_in_A))).mean() / 2

        naive_loss = (1 - criterion(target_features_in_A, converted_target_features_in_A).mean()) / 2
        loss = (same_loss + naive_loss) / 2
        # logger.info(f'same_loss {same_loss}, naive_loss {naive_loss}, loss {loss}')
        

        # Log a loss history
        loss_history = np.append(loss_history, torch.clone(loss).detach().cpu().numpy())

        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            logger.info(f"[Train] [Loss {loss.item():.8f}] [{i * len(images):5d}/{data_num:5d}]")

    loss_avg = loss_history.mean()

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
    loss_history = np.array([])

    AE.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            loss = 0
            images = images.to(device)

            target_features_T = T(transform_T(images))
            converted_target_features_in_A = AE(target_features_T)
            target_features_in_A = A(transform_A(images))
            loss = (1 - criterion(target_features_in_A, converted_target_features_in_A).mean()) / 2

        loss_history = np.append(loss_history, torch.clone(loss).detach().cpu().numpy())

    loss_avg = loss_history.mean()

    logger.info(f"[{mode} avg] [Loss {loss_avg:.10f} (avg)]")

    return loss_avg.item()

def set_global():
    global options
    global device

    options = get_options()
    # Decide device
    gpu_id = get_freer_gpu()
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    # Decide device
    # device = f"cuda:{options.gpu_idx}" if torch.cuda.is_available() else "cpu"

    options.device = device

def main():

    set_global()

    if options.resume:
        if options.resume_epoch is None:
            print('Specify resume_epoch')
            exit(1)

    # Create directory to save results
    base_result_dir = resolve_path(options.result_dir, (options.identifier + f'_{options.num_of_identities}_identities'))

    if os.path.exists(base_result_dir):
        base_result_dir = base_result_dir + str(randint(0, 1e9))
        os.makedirs(base_result_dir, exist_ok=False)
    else:
        os.makedirs(base_result_dir, exist_ok=False)
    
    # Save options by json format
    save_json(resolve_path(base_result_dir, "step1.json"), vars(options))

    # Create logger
    logger = create_logger(f"Step 1", resolve_path(base_result_dir, "training.log"))

    # Log options
    logger.info(vars(options))

    # Load models
    img_size_T = get_img_size(options.target_model)
    img_size_A = get_img_size(options.attack_model)

    log_test_loss = np.array([])

    for i_iter in range(options.iteration):

        AE = load_autoencoder(
            pretrained=options.resume,
            model_path=options.AE_path,
            mode='train',
            ver=options.AE_ver
        )
        T, _ = load_model_as_feature_extractor(
            arch=options.target_model,
            embedding_size=options.target_embedding_size,
            mode='eval',
            path=options.target_model_path,
            pretrained=True
        )
        A, _ = load_model_as_feature_extractor(
            arch=options.attack_model,
            embedding_size=options.attack_embedding_size,
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

        # Prepare dataset
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
            batch_size=options.num_per_identity,
            shuffle=False,
            # Optimization:
            num_workers=os.cpu_count(),
            pin_memory=True
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=options.num_per_identity,
            shuffle=True,
            # Optimization:
            num_workers=os.cpu_count(),
            pin_memory=True
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=options.num_per_identity,
            shuffle=False,
            # Optimization:
            num_workers=os.cpu_count(),
            pin_memory=True
        )

        # Prepare transfomer
        transform_T = transforms.Compose([
            transforms.Resize((img_size_T, img_size_T)),
        ])
        transform_A = transforms.Compose([
            transforms.Resize((img_size_A, img_size_A)),
        ])

        # Prepare optimizer and criterion
        optimizer = optim.Adam(AE.parameters(), lr=options.learning_rate, betas=(options.beta1, options.beta2), weight_decay=options.weight_decay)
        criterion = nn.CosineSimilarity()

        result_dir = resolve_path(base_result_dir, str(i_iter))
        os.makedirs(result_dir, exist_ok=False)

        # Initialize variables for training
        best_model_wts = copy.deepcopy(AE.state_dict())
        best_loss = 1e9
        best_loss_test = 1e9
        loss_update_counter = 0

        # Set resume
        if options.resume:
            start_epoch = options.resume_epoch
        else:
            start_epoch = 0

        ##########################
        ### Converter Training ###
        ##########################

        # Start training
        for epoch in range(start_epoch + 1, options.n_epochs + 1):
            if loss_update_counter > options.early_stop:
                log_test_loss = np.append(log_test_loss, best_loss_test)
                break
            logger.info(f"[{i_iter}] [Epoch {epoch:d}]")
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

    logger.info(f'Average test loss is {np.average(log_test_loss)}')
        
if __name__ == '__main__':
    main()