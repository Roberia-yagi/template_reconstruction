import sys
sys.path.append("../")

import argparse
import os
import numpy as np
import math
import time
import datetime
import json
from typing import Any, Union, Tuple, Dict, List

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torch.autograd import grad
from torchvision import datasets
from torch.autograd import Variable

import torch
from torch import nn
from torch import optim
from torch import linalg as LA

from models.Discriminator3 import Discriminator3
from models.Generator3 import Generator3

import matplotlib.pyplot as plt

from util import (save_json, load_json, create_logger, resolve_path,
    load_classifier, load_classifier_as_feature_extractor)
from celeba import CelebA

def get_options() -> Any:
    parser = argparse.ArgumentParser()

    # Timestamp
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    parser.add_argument("--identifier", type=str, default=time_stamp, help="identifier")

    # Save
    parser.add_argument("--save", type=bool, default=False, help="if save log and data")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")

    # Directories
    parser.add_argument("--dataset_dir", type=str, default="~/nas/dataset/CelebA_MTCNN160", help="path to directory which includes dataset(CelebA)")
    parser.add_argument("--result_dir", type=str, default="~/nas/results/step2", help="path to directory which includes results")
    parser.add_argument("--step1_dir", type=str, default="facenet_mtcnn", help="path to directory which includes the step1 result")

    # Method type
    parser.add_argument("--method", type=str, default="pure", help="method type: 'pure', 'existing', 'proposed'")

    # For training
    parser.add_argument("--gpu_idx", type=int, default=0, help="index of cuda devices")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--learning_rate", type=float, default=0.0002, help="learning rate")
    parser.add_argument("--alpha", type=float, default=0.5, help="alpha")
    parser.add_argument("--beta", type=float, default=0.5, help="beta")

    # For WGAN
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")

    # Conditions
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--G_network_dim", type=int, default=64, help="unit dimensionality of G")
    parser.add_argument("--D_network_dim", type=int, default=64, help="unit dimensionality of D")
    parser.add_argument("--img_crop", type=int, default=100, help="size of cropping image")
    parser.add_argument("--img_channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--target_model", type=str, default="VGG16", help="target model(classifier): 'VGG16', 'ResNet152', 'FaceNet'")
    parser.add_argument("--target_index", type=int, default=0, help="index of target user on target model(classifier)")

    opt = parser.parse_args()

    return opt

def override_options(options: argparse.Namespace, options_path: str) -> Tuple[argparse.Namespace, int, bool]:
    if not os.path.exists(options_path):
        return options, 0, False

    options_json = load_json(options_path)
    overrided_options = argparse.Namespace()

    # Clone options
    for k, v in options_json.items():
        setattr(overrided_options, k, v)

    last_epoch = overrided_options.n_epochs

    overrided_options.save = options.save
    overrided_options.gpu_idx = options.gpu_idx
    overrided_options.n_epochs += options.n_epochs

    return overrided_options, last_epoch, True

# For WGAN with GP
# Learn the algorithm and rewrite
def calculate_gradient_penalty(D: nn.Module, x: torch.Tensor.data, y:torch.Tensor.data, device: str) -> float:
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).to(device)
    z = x + alpha * (y - x)
    z = z.to(device)
    z.requires_grad = True

    o = D(z)
    gradient = grad(o, z, grad_outputs = torch.ones(o.size()).to(device), create_graph = True)[0].view(z.size(0), -1)
    gradient_penalty = ((gradient.norm(p = 2, dim = 1) - 1) ** 2).mean()

    return gradient_penalty

def train_D(D: nn.Module, G: nn.Module, optimizer_D: optim.Optimizer, device: str,
            batch_size: int, latent_dim: int, clip_value: float, x: torch.Tensor) -> Tuple[float, float, float]:

    optimizer_D.zero_grad()

    z = torch.randn(batch_size, latent_dim).to(device)
    Gz = G(z).detach()

    loss_Dx = -torch.mean(D(x))
    loss_DGz = torch.mean(D(Gz))
    loss_D = loss_Dx + loss_DGz
    gradient_penalty = calculate_gradient_penalty(D, x.data, Gz.data, device)
    loss_D = loss_D + gradient_penalty * 10.0
    loss_D.backward()

    optimizer_D.step()

    # for p in D.parameters():
    #     p.data.clamp_(-clip_value, clip_value)

    return loss_D.item(), loss_Dx.item(), loss_DGz.item()

def train_G(D: nn.Module, G: nn.Module, optimizer_G: optim.Optimizer, device: str,
            batch_size: int, latent_dim: int) -> Tuple[torch.Tensor, float]:
 
    optimizer_G.zero_grad()

    z = torch.randn(batch_size, latent_dim).to(device)
    Gz = G(z)

    loss_G = -torch.mean(D(Gz))
    loss_G.backward()

    optimizer_G.step()

    return Gz, loss_G.item()

def main():
    # Get options
    options = get_options()

    # Override options if options json exists
    result_dir = resolve_path(options.result_dir, options.identifier)
    options_path = resolve_path(result_dir,"step2.json")
    options, base_epoch, hasOverrided = override_options(
        options,
        options_path
    )

    print(options)

    # Decide device
    device = f"cuda:{options.gpu_idx}" if torch.cuda.is_available() else "cpu"
    options.device = device

    # Load step1 options
    step1_dir = "~/nas/results/step1/" + options.step1_dir
    step1_options = load_json(resolve_path(step1_dir, "step1.json"))
    label_map = load_json(resolve_path(step1_dir, "label_map.json"))

    # Create directories to save data
    if options.save:
        result_images_dir = resolve_path(result_dir, "images")

        # Create directory to save results
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(result_images_dir, exist_ok=True)

        # Save step1 options
        save_json(resolve_path(result_dir, "step1.json"), step1_options)
        save_json(resolve_path(result_dir, "label_map.json"), label_map)

        # Save options by json format
        save_json(resolve_path(result_dir, "step2.json"), vars(options))

    # Create private identity set
    private_identity_set = set([int(label) + 1 for label in label_map.keys()])

    # Create logger
    if options.save:
        logger = create_logger(f"Step 2 ({options.method} method)", resolve_path(result_dir, "training.log"))
    else:
        logger = create_logger(f"Step 2 ({options.method} method)")
    
    # Log options
    logger.info(vars(options))

    # Create data loader
    dataset = CelebA(
        base_dir=options.dataset_dir,
        usage='all',
        exclude=private_identity_set,
        transform=transforms.Compose([
            # transforms.CenterCrop(options.img_crop),
            # transforms.Resize((options.img_size, options.img_size)),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, ), (0.5, ))
        ])
    )
    dataloader = DataLoader(
        dataset,
        batch_size=options.batch_size,
        shuffle=True,
        # Optimization:
        num_workers=os.cpu_count(),
        pin_memory=True
    )

    # Initialize generator and discriminator
    img_shape = (options.img_channels, options.img_size, options.img_size)
    D = Discriminator3(
        input_dim=3,
        network_dim=options.D_network_dim,
        img_shape=img_shape
    ).to(device)
    G = Generator3(
        latent_dim=options.latent_dim,
        network_dim=options.G_network_dim,
        img_shape=img_shape
    ).to(device)

    if hasOverrided:
        D.load_state_dict(torch.load(resolve_path(result_dir, "D.pth")))
        G.load_state_dict(torch.load(resolve_path(result_dir, "G.pth")))

    
    # Optimize performance
    if device == 'cuda':
        '''
        D = torch.nn.DataParallel(D)
        G = torch.nn.DataParallel(G)
        '''
        torch.backends.cudnn.benchmark = True

    # Optimizers
    optimizer_D = torch.optim.Adam(D.parameters(), lr=options.learning_rate, betas=(0.5, 0.999))
    optimizer_G = torch.optim.Adam(G.parameters(), lr=options.learning_rate, betas=(0.5, 0.999))

    # ----------
    #  Training
    # ----------

    for epoch in range(base_epoch + 1, options.n_epochs + 1):
        epoch_start_time = time.time()

        for i, (imgs, _) in enumerate(dataloader):
            # Configure input
            x = imgs.to(device)
            batch_size = x.shape[0]


            # ---------------------
            #  Train Discriminator
            # ---------------------
            loss_D, loss_Dx, loss_DGz = train_D(D, G, optimizer_D, device, batch_size,
                                                options.latent_dim, options.clip_value, x)
            
            # Train the generator every n_critic iterations
            if i % options.n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------
                if options.method == 'pure':
                    Gz, loss_G = train_G(D, G, optimizer_G, device, batch_size, options.latent_dim)
                    logger.info(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, Dx: %f, Gz: %f] [G loss: %f]" % (
                            epoch, options.n_epochs, i + 1, len(dataloader),
                            loss_D, loss_Dx, loss_DGz, loss_G
                        )
                    )
            
        if options.save:
            z = torch.randn(
                options.batch_size,
                options.latent_dim,
            ).to(device)
            images = G(z)

            # denormalized = images.mul(0.5).add(0.5)

            image_path = resolve_path(
                result_images_dir,
                f"epoch{epoch}.png"
            )

            save_image(
                images,
                image_path,
                nrow=int(options.batch_size ** (1/2)),
                normalize=True
            )

            logger.info(f"[Saved image: {image_path}]")

        epoch_elapsed_time = time.time() - epoch_start_time
        estimated_rest_time = epoch_elapsed_time * (options.n_epochs - base_epoch + epoch)
        estimated_finish_time = datetime.datetime.now() + datetime.timedelta(seconds=estimated_rest_time)
        logger.debug(f"[Epcoh {epoch} elapsed time: {epoch_elapsed_time} estimated finish time: {estimated_finish_time}]")

    if options.save:
        torch.save(D.state_dict(), resolve_path(result_dir, "D.pth"))
        torch.save(G.state_dict(), resolve_path(result_dir, "G.pth"))

if __name__ == '__main__':
    main()