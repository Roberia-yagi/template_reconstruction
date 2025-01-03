# Train converter
import pickle
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
from tqdm import tqdm 
from typing import Any, Tuple

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.casia_web_face_dual import CasiaWebFaceDual

from utils.util import (save_json, create_logger, resolve_path, load_model_as_feature_extractor,
    get_img_size, load_autoencoder, set_global)

def get_options() -> Any:
    parser = argparse.ArgumentParser()

    # Timestamp
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    parser.add_argument("--identifier", type=str, default=time_stamp, help="timestamp")
    parser.add_argument("--gpu_idx", type=int, default=None, help="index of cuda devices")

    # Save
    parser.add_argument("--log_interval", type=int, default=100, help="interval between logs (per iteration)")

    # Directories
    parser.add_argument("--result_dir", type=str, default="../../../results/train_transferer_cossim", help="path to directory which includes results")
    parser.add_argument("--target_dataset_dir", type=str, required=True)
    parser.add_argument("--attack_dataset_dir", type=str, required=True)
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
    parser.add_argument("--target_model", type=str, required=True, help="target model: 'FaceNet', 'Arcface', 'Magface")
    parser.add_argument("--attack_model", type=str, required=True, help="attack model: 'FaceNet', 'Arcface', 'Magface")
    parser.add_argument("--AE_ver", type=float, default=1.1, help="AE version: 1, 1.1, 1.2, 1.3, 1.4")

    # Conditions of training
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--n_epochs", type=int, default=int(1e9), help="number of epochs of training")
    parser.add_argument("--resume", action='store_true', help='flag of resume')
    parser.add_argument("--resume_epoch", type=int, help='flag of resume')
    parser.add_argument("--gamma", type=float, default=1, help='weight of negative loss')
    parser.add_argument("--num_of_identities", type=int, required=True, help="Number of unique identities")
    parser.add_argument("--num_per_identity", type=int, required=True, help="Number of unique identities")
    parser.add_argument("--early_stop", type=int, default=1, help="the trial limitation of non-update training")
    parser.add_argument("--negative_loss", action='store_true', help='flag of negative loss')
    parser.add_argument("--num_of_samples", type=int, default=10, help="Number of samples for calculation of negative loss")
    parser.add_argument("--iteration", type=int, default=5, help="the number of training for fair evaluation")

    opt = parser.parse_args()

    return opt


def train_target(
    dataloader: DataLoader,
    T: nn.Module,
    A: nn.Module,
    AE: nn.Module,
    all_features_labels: tuple,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    logger: Logger,
    log_interval: int
) -> Tuple[float, float]:
    data_num = len(dataloader.dataset)
    loss_history = np.array([])
    loss_same_history = np.array([])
    loss_diff_history = np.array([])

    AE.train()
    for i, ((images_t, images_a), labels) in enumerate(dataloader):
        images_t = images_t.to(device)
        images_a = images_a.to(device)

        optimizer.zero_grad()

        # Extract features 
        target_features_T = T(images_t)
        
        # Convert features in the target model space to in the attack model space
        converted_target_features_in_A = AE(target_features_T)
        target_features_in_A = A(images_a)
        loss_same = (1 - criterion(target_features_in_A, converted_target_features_in_A).mean()) / 2
        loss_diff = torch.tensor(0.0, dtype=float).to(device)
        loss = loss_same / 2

        if options.negative_loss:
            for converted_target_feature_in_A, target_feature_A, target_label in zip(converted_target_features_in_A, target_features_in_A, labels):
                other_features, other_labels = all_features_labels
                trimmed_other_features = other_features[other_labels != target_label]

                # Sample other features
                weights = torch.tensor(np.full(trimmed_other_features.shape[0], 1/trimmed_other_features.shape[0]), dtype=torch.float)
                index = weights.multinomial(num_samples=options.num_of_samples, replacement=True)
                trimmed_other_features = trimmed_other_features[index]

                expanded_converted_target_feature_in_A = converted_target_feature_in_A.expand(trimmed_other_features.shape[0], -1)
                expanded_target_feature_A = target_feature_A.expand(trimmed_other_features.shape[0], -1)
                converted_cossim = criterion(expanded_converted_target_feature_in_A, trimmed_other_features.to(device))
                cossim = criterion(expanded_target_feature_A, trimmed_other_features.to(device))
                loss_diff += torch.abs((converted_cossim - cossim) / 2).mean()
            loss_diff /= len(labels)

            loss = (loss_same + options.gamma * loss_diff) / 2

        # Log a loss history
        loss_history = np.append(loss_history, torch.clone(loss).detach().cpu().numpy())
        loss_same_history = np.append(loss_same_history, torch.clone(loss_same).detach().cpu().numpy())
        loss_diff_history = np.append(loss_diff_history, torch.clone(loss_diff).detach().cpu().numpy())

        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            logger.info(f"[Train] [Loss {loss.item():.8f}] [Same Loss {loss_same.item():.8f}] [Diff Loss {loss_diff.item():.8f}] [{i * len(images_t):5d}/{data_num:5d}]")

    loss_avg = loss_history.mean()
    loss_same_avg = loss_same_history.mean()
    loss_diff_avg = loss_diff_history.mean()

    logger.info(f"[Train avg] [Loss {loss_avg:.10f} (avg)] [Same Loss {loss_same_avg:.10f} (avg)] [Diff Loss {loss_diff_avg:.10f} (avg)]")

    return loss_avg.item()


def eval_target(
    mode: str,
    dataloader: DataLoader,
    T: nn.Module,
    A: nn.Module,
    AE: nn.Module,
    all_features_labels: tuple,
    criterion: nn.Module,
    logger: Logger
) -> Tuple[float, float]:
    data_num = len(dataloader.dataset)
    loss_history = np.array([])
    loss_same_history = np.array([])
    loss_diff_history = np.array([])

    AE.eval()
    with torch.no_grad():
        for i, ((images_t, images_a), labels) in enumerate(dataloader):
            images_t = images_t.to(device)
            images_a = images_a.to(device)

            target_features_T = T(images_t)
            converted_target_features_in_A = AE(target_features_T)
            target_features_A = A(images_a)
            loss_same = (1 - criterion(target_features_A, converted_target_features_in_A).mean()) / 2
            loss_diff = torch.tensor(0.0, dtype=float).to(device)
            loss = loss_same

            if options.negative_loss:
                for converted_target_feature_in_A, target_feature_A, target_label in zip(converted_target_features_in_A, target_features_A, labels):
                    other_features, other_labels = all_features_labels
                    trimmed_other_features = other_features[other_labels != target_label]

                    # Sample other features
                    weights = torch.tensor(np.full(trimmed_other_features.shape[0], 1/trimmed_other_features.shape[0]), dtype=torch.float)
                    index = weights.multinomial(num_samples=options.num_of_samples, replacement=True)
                    trimmed_other_features = trimmed_other_features[index]

                    expanded_converted_target_feature_in_A = converted_target_feature_in_A.expand(trimmed_other_features.shape[0], -1)
                    expanded_target_feature_A = target_feature_A.expand(trimmed_other_features.shape[0], -1)
                    converted_cossim = criterion(expanded_converted_target_feature_in_A, trimmed_other_features.to(device))
                    cossim = criterion(expanded_target_feature_A, trimmed_other_features.to(device))
                    loss_diff += torch.abs((converted_cossim - cossim) / 2).mean()
                loss_diff /= len(labels)

            loss = (loss_same + loss_diff) / 2
        loss_history = np.append(loss_history, torch.clone(loss).detach().cpu().numpy())
        loss_same_history = np.append(loss_same_history, torch.clone(loss_same).detach().cpu().numpy())
        loss_diff_history = np.append(loss_diff_history, torch.clone(loss_diff).detach().cpu().numpy())

    loss_avg = loss_history.mean()
    loss_same_avg = loss_same_history.mean()
    loss_diff_avg = loss_diff_history.mean()

    logger.info(f"[{mode} avg] [Loss {loss_avg:.10f} (avg)] [Same Loss {loss_same_avg:.10f} (avg)] [Diff Loss {loss_diff_avg:.10f} (avg)]")

    return loss_avg.item()

def calculate_features_from_dataset(model, dataloader, usage):
    all_features = torch.tensor([])
    all_labels = np.array([])
    dataset_name = options.dataset_dir[options.dataset_dir.rfind('/')+1:]
    pkl_path = resolve_path(f'{options.result_dir}_negative_loss', f'{usage}_{dataset_name}_{options.attack_model}_{options.num_of_identities}_{options.num_per_identity}_identities.pkl')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            all_features, all_labels = pickle.load(f)
            print(f'{usage} features are loaded by pickle')
    else:
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            features = model(images).detach().cpu()
            all_features = torch.concat((all_features, features))
            all_labels = np.append(all_labels, labels)
        with open(pkl_path, 'wb') as f:
            pickle.dump((all_features, all_labels), f)
            print(f'{usage} features are calculated and saved as pickle')

    return all_features,all_labels 


def main():
    global options
    global device
    device, options = set_global(get_options)

    if options.resume:
        if options.resume_epoch is None:
            print('Specify resume_epoch')
            exit(1)

    # Create directory to save results
    if options.negative_loss:
        base_result_dir = resolve_path(f'{options.result_dir}_negative_loss', (options.identifier + f'_{options.num_of_identities}_identities'))
    else:
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
    log_test_loss = np.array([])

    for i_iter in range(options.iteration):

        AE = load_autoencoder(
            pretrained=options.resume,
            model_path=options.AE_path,
            mode='train', ver=options.AE_ver
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
        train_dataset = CasiaWebFaceDual(base_dir1=options.target_dataset_dir,
                                        base_dir2=options.attack_dataset_dir,
                                        usage='train',
                                        num_of_identities=options.num_of_identities,
                                        num_per_identity=options.num_per_identity,
                                        transform=transforms.ToTensor())
        test_dataset = CasiaWebFaceDual(base_dir1=options.target_dataset_dir,
                                        base_dir2=options.attack_dataset_dir,
                                        usage='test',
                                        num_of_identities=options.num_of_identities,
                                        num_per_identity=options.num_per_identity,
                                        transform=transforms.ToTensor())
        val_dataset = CasiaWebFaceDual(base_dir1=options.target_dataset_dir,
                                        base_dir2=options.attack_dataset_dir,
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

        # Prepare optimizer and criterion
        optimizer = optim.Adam(AE.parameters(), lr=options.learning_rate, betas=(options.beta1, options.beta2), weight_decay=options.weight_decay)
        criterion = nn.CosineSimilarity()

        result_dir = resolve_path(base_result_dir, str(i_iter))
        os.makedirs(result_dir, exist_ok=False)

        # Calculate features for negative loss calculation
        if options.negative_loss:
            features_train, labels_train = calculate_features_from_dataset(A, train_dataloader, 'train')
            features_test, labels_test= calculate_features_from_dataset(A, test_dataloader, 'test')
            features_val, labels_val= calculate_features_from_dataset(A, val_dataloader, 'valid')
        else:
            features_train, labels_train = None, None
            features_test, labels_test= None, None
            features_val, labels_val= None, None

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
                AE,
                (features_train, labels_train),
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
                AE,
                (features_val, labels_val),
                criterion,
                logger
            )

            # Test model
            test_loss= eval_target(
                "Test",
                test_dataloader,
                T,
                A,
                AE,
                (features_test, labels_test),
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