import sys
sys.path.append("../")

import os
import argparse
import time
import datetime
import copy
import json
from logging import Logger
from collections import Counter
from typing import Any, Union, Tuple, Dict, List

import torch
from torch import nn
from torch import optim
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torchvision.models as models

from facenet_pytorch import InceptionResnetV1
from sklearn.model_selection import train_test_split

from util import save_json, create_logger, resolve_path
from celeba import CelebA

def load_target_model(model: str, classifier_mode:str, num_classes: int) -> Union[nn.Module, None]:
    if model == "VGG16":
        return models.vgg16(num_classes=num_classes, pretrained=True) # pre-trained on ImageNet
        # return models.vgg16(num_classes=num_classes)

    if model == "ResNet152":
        return models.resnet152(num_classes=num_classes, pretrained=True) # pre-trained on ImageNet
        # return models.resnet152(num_classes=num_classes)
    
    if model == "FaceNet":
        if classifier_mode == "multi_class":
            # return InceptionResnetV1(classify=True,num_classes=num_classes)
            return InceptionResnetV1(classify=True, num_classes=num_classes, pretrained="vggface2")
        elif classifier_mode == "features":
            return InceptionResnetV1(classify=False, num_classes=None, pretrained="vggface2")

    return None

def count(arr: List[torch.Tensor]) -> Counter:
    c = Counter()

    for x in arr:
        c[x.item()] += 1
    
    return c

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
        transforms.ToTensor()
        # transforms.Normalize(
        #     mean=(0.5,), std=(0.5,)
            # mean=[0.6327, 0.4673, 0.3879], std=[0.2486, 0.2164, 0.2053] # All classes (by Dr.Ohki)
            # mean=[0.519415020942688, 0.43464329838752747, 0.3883533179759979], std=[0.2693840563297272, 0.2513585090637207, 0.2500663697719574], # 1000 classes
        # )
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

def train_target(
    dataloader: DataLoader,
    model: nn.Module,
    device: str,
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
    device: str,
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

def compute_mean_std(
    dataloader: DataLoader
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    images_mean = torch.tensor([0.0, 0.0, 0.0])
    images_var = torch.tensor([0.0, 0.0, 0.0])

    for images, _ in dataloader:
        images_mean += images.mean([2, 3]).sum(0)
        images_var += images.var([2, 3]).sum(0)
    
    images_mean /= len(dataloader.dataset)
    images_var /= len(dataloader.dataset)

    mean = [x.item() for x in images_mean]
    std = [torch.sqrt(x).item() for x in images_var]

    return mean, std

def getOptions() -> Any:
    parser = argparse.ArgumentParser()

    # Timestamp
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    parser.add_argument("--identifier", type=str, default=time_stamp, help="timestamp")

    # Save
    parser.add_argument("--save", type=bool, default=False, help="if save log and data")
    parser.add_argument("--log_interval", type=int, default=100, help="interval between logs (per iteration)")

    # Mode
    parser.add_argument("--classifier_mode", type=str, default="multi_class", help="classifier_mode: 'multi_class', 'features'")

    # Directories
    parser.add_argument("--dataset_dir", type=str, default="~/nas/dataset/CelebA_MTCNN160", help="path to directory which includes dataset(CelebA)")
    parser.add_argument("--result_dir", type=str, default="~/nas/results/step1", help="path to directory which includes results")

    # For training
    parser.add_argument("--gpu_idx", type=int, default=0, help="index of cuda devices")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate of optimizer")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum of optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="learning rate of optimizer")

    # Conditions
    parser.add_argument("--img_crop", type=int, default=100, help="size of cropping image")
    parser.add_argument("--img_size", type=int, default=160, help="size of image")
    parser.add_argument("--test_size", type=float, default=0.1, help="ratio of test data to total data")
    parser.add_argument("--val_size", type=float, default=0.1, help="ratio of validation data to total data")
    parser.add_argument("--target_model", type=str, default="VGG16", help="target model(classifier): 'VGG16', 'ResNet152', 'FaceNet'")

    opt = parser.parse_args()

    return opt

def main():
    num_of_identities = 1000
    num_per_identity = 30

    # Get options
    options = getOptions()

    # Decide device
    device = f"cuda:{options.gpu_idx}" if torch.cuda.is_available() else "cpu"
    options.device = device

    # Create directories to save data
    if options.save:
        result_dir = resolve_path(options.result_dir, options.identifier)

        # Create directory to save result
        os.makedirs(result_dir, exist_ok=True)

        # Save options by json format
        save_json(resolve_path(result_dir, "step1.json"), vars(options))
    
    # Create logger
    if options.save:
        logger = create_logger(f"Step 1 ({options.target_model})", resolve_path(result_dir, "training.log"))
    else:
        logger = create_logger(f"Step 1")
    
    # Load private datasets
    training_data, test_data, val_data, label_map = load_celeba_private_datasets(
        base_dir=options.dataset_dir,
        select=(num_of_identities, num_per_identity),
        test_size=options.test_size, 
        val_size=options.val_size, 
        img_crop_size=options.img_crop,
        img_size=options.img_size
    )
    train_dataloader = DataLoader(
        training_data,
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

    # mean, std = compute_mean_std(train_dataloader)

    # Save label mappings
    if options.save:
        # Save label mappings by json format
        save_json(resolve_path(result_dir, "label_map.json"), label_map)

    # Load target model
    model = load_target_model(
        options.target_model,
        options.classifier_mode,
        num_classes=num_of_identities
    ).to(device)

    # Create optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=options.learning_rate,
        momentum=options.momentum,
        weight_decay=options.weight_decay
    )

    # Create loss function
    if options.classifier_mode == "multi_class":
        criterion = nn.CrossEntropyLoss()
    elif options.classifier_mode == "features":
        criterion = nn.CosineSimilarity()

    # Train, valid, test model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_acc_test = 0.0
    for epoch in range(1, options.n_epochs + 1):
        logger.info(f"[Epoch {epoch:d}/{options.n_epochs:d}]")
        epoch_start_time = time.time()

        # Load best model
        model.load_state_dict(best_model_wts)

        # Train model
        train_loss, train_acc = train_target(
            train_dataloader,
            model,
            device,
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
            device,
            criterion,
            label_map,
            logger
        )

        # Test model
        test_loss, test_acc = eval_target(
            "Test",
            test_dataloader,
            model,
            device,
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
        logger.info(f"[Best model: Test acc {test_acc*100:>0.1f}%]")

    if options.save:
        model_path = resolve_path(
            result_dir,
            f"{options.target_model}.pth"
        )
        torch.save(model.state_dict(), model_path)
        logger.info(f"[Saved model: {model_path}]")

if __name__ == '__main__':
    main()
