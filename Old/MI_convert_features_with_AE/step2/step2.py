# See if AE correctly recover image with GAN
import sys

sys.path.append("../")

import os
import argparse
import glob
import time
import datetime
import PIL
from typing import Any, List, Tuple

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image


from util import (save_json, load_json, create_logger, resolve_path, load_model_as_feature_extractor, RandomBatchLoader,
    load_attacker_discriminator, load_attacker_generator, get_freer_gpu, get_img_size)

from celeba import CelebA


def get_options() -> Any:
    parser = argparse.ArgumentParser()

    # Timestamp
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    parser.add_argument("--identifier", type=str, default=time_stamp, help="timestamp")

    # Save
    parser.add_argument("--save", type=bool, default=True, help="if save log and data")

    # Directories
    parser.add_argument("--result_dir", type=str, default="~/nas/results/MI_with_DA/step3", help="path to directory which includes results")
    parser.add_argument("--step2_dir", type=str, default="pure_facenet_500epoch_features", help="path to directory which includes the step2 result")
    parser.add_argument("--target_image_dir", type=str, default="~/nas/dataset/target_images", help="path to directory which contains target images")

    # For inference
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--times", type=int, default=5, help="times to initialize z")
    parser.add_argument("--epochs", type=int, default=1500, help="times to initialize z") 
    parser.add_argument("--iterations", type=int, default=8, help="iterations to optimize z")
    parser.add_argument("--learning_rate", type=float, default=0.02, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="learning rate")
    parser.add_argument("--lambda_i", type=float, default=100, help="learning rate")

    # Conditions
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--target_model", type=str, default="FaceNet", help="target model: 'FaceNet', 'Arcface'")
    parser.add_argument("--attack_model", type=str, default="Arcface", help="attack model: 'FaceNet', 'Arcface'")
    parser.add_argument("--target_dir_name", type=str, default="0", help="directory name which contains images of target user(features)")
    parser.add_argument("--single_mode", type=bool, default=True, help="True if you use just one stored feature(features)")

    opt = parser.parse_args()

    return opt

def L_prior(D: nn.Module, G: nn.Module, z: torch.Tensor) -> torch.Tensor:
    return torch.mean(-D(G(z)))


def calc_id_loss(G: nn.Module, FE: nn.Module, z: torch.Tensor, device: str, all_target_features: torch.Tensor, image_size: int) -> torch.Tensor:
    resize = transforms.Resize((image_size, image_size))
    metric = nn.CosineSimilarity(dim=1)
    Gz_features = FE(resize(G(z))).to(device)
    dim = Gz_features.shape[1]

    sum_of_cosine_similarity = 0
    for target_feature in all_target_features.view(-1, dim):
        target_feature = target_feature.expand(Gz_features.shape[0], -1)
        sum_of_cosine_similarity += metric(target_feature, Gz_features)
    return 1 - torch.mean(sum_of_cosine_similarity / all_target_features.shape[0])

def get_best_image(FE: nn.Module, images: nn.Module, image_size: int, all_target_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    resize = transforms.Resize((image_size, image_size))
    metric = nn.CosineSimilarity(dim=1)
    FEz_features = FE(resize(images))
    dim = FEz_features.shape[1]
    sum_of_cosine_similarity = 0
    for target_feature in all_target_features.view(-1, dim):
        target_feature = target_feature.expand(FEz_features.shape[0], -1)
        sum_of_cosine_similarity += metric(FEz_features, target_feature)
    sum_of_cosine_similarity /= all_target_features.shape[0]
    bestImageIndex = sum_of_cosine_similarity.argmax()
    return images[bestImageIndex], sum_of_cosine_similarity[bestImageIndex]

options = get_options()

def main():

    # Decide device
    gpu_id = get_freer_gpu()
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    options.device = device

    # Load step1 and step2 options
    step2_dir = "~/nas/results/common/step2/" + options.step2_dir
    step1_options = load_json(resolve_path(step2_dir, "step1.json"))
    step2_options = load_json(resolve_path(step2_dir, "step2.json"))
    label_map = load_json(resolve_path(step2_dir, "label_map.json"))

    # Set step1 dir
    step1_dir = "~/nas/results/step1/" + step2_options["step1_dir"]


    # Create directories to save data
    if options.save:
        # Create directory to save results
        result_dir = resolve_path(options.result_dir, options.identifier)
        os.makedirs(result_dir, exist_ok=True)
        
        # Create directory to save each result
        for i in range(1, options.times + 1):
            result_each_time_dir = resolve_path(result_dir, f"{i}")
            os.makedirs(result_each_time_dir, exist_ok=True)

        # Save step1 and step2 options
        save_json(resolve_path(result_dir, "step1.json"), step1_options)
        save_json(resolve_path(result_dir, "step2.json"), step2_options)

        # Save options by json format
        save_json(resolve_path(result_dir, "step3.json"), vars(options))

    # Create logger
    if options.save:
        logger = create_logger(f"Step 3", resolve_path(result_dir, "training.log"))
    else:
        logger = create_logger(f"Step 3")

    # Log options
    logger.info(vars(options))

    # Load models
    img_size = get_img_size(options.target_model)
    img_shape = (options.img_channels, img_size, img_size)

    D = load_attacker_discriminator(
        path=resolve_path(step2_dir, "D.pth"),
        input_dim=step2_options["img_channels"],
        network_dim=step2_options["D_network_dim"],
        img_shape=img_shape
    ).to(device)
    G = load_attacker_generator(
        path=resolve_path(step2_dir, "G.pth"),
        latent_dim=options.latent_dim,
        network_dim=step2_options["G_network_dim"],
        img_shape=img_shape
    ).to(device)
    T = load_model_as_feature_extractor(
        model=options.target_model
    ).to(device)
    A = load_model_as_feature_extractor(
        model=options.attack_model
    ).to(device)
    
    D.eval()
    G.eval()
    T.eval()
    A.eval()

    # Create data loader
    dataset = CelebA(
        base_dir=step2_options["dataset_dir"],
        usage='all',
        exclude=None,
        transform=transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
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

    target_imagefolder_path = resolve_path(options.target_image_dir, options.target_dir_name)
    all_target_images = torch.tensor([]).to(device)
    all_target_features = torch.tensor([]).to(device)
    transform=transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    # Convert all taget images in target imagefolder to features
    for filename in glob.glob(target_imagefolder_path + "/*.jpg"):
        target_image = PIL.Image.open(filename)
        converted_target_image = transform(target_image).to(device)
        all_target_images= torch.cat((all_target_images, converted_target_image.unsqueeze(0)))

        target_feature = T(converted_target_image.view(1, -1, img_size, img_size)).detach().to(device)
        all_target_features= torch.cat((all_target_features, target_feature.unsqueeze(0)))

        if options.single_mode:
            break


    # Search z^
    for i in range(1, options.times + 1):
        z = torch.randn(options.batch_size * options.iterations, options.latent_dim, requires_grad=True, device=device)
        optimizer = optim.SGD([z], lr=options.learning_rate, momentum=options.momentum)
        dataloader = RandomBatchLoader(z, options.batch_size)

        # Optimize z
        start_time = time.time()
        for epoch in range(1, options.epochs + 1):
            L_prior_loss_avg, L_id_loss_avg, total_loss_avg = 0, 0, 0

            for batch_idx, batch in enumerate(dataloader):
                optimizer.zero_grad()

                L_prior_loss = L_prior(D, G, batch)
                L_id_loss = calc_id_loss(G, A, batch, device, all_target_features, img_size) 
                
                L_id_loss = options.lambda_i * L_id_loss
                total_loss = L_prior_loss + L_id_loss

                L_prior_loss_avg += L_prior_loss
                L_id_loss_avg += L_id_loss
                total_loss_avg += total_loss

                total_loss.backward()
                optimizer.step()

            L_prior_loss_avg /= z.shape[0]
            L_id_loss_avg /= z.shape[0]
            total_loss_avg /= z.shape[0]

            logger.info(f"[{i}/{options.times}], \
                        [{epoch}/{options.epochs}],\
                        [Loss: {total_loss_avg.item()},\
                        L_prior: {L_prior_loss_avg.item()},\
                        L_id: {L_id_loss_avg.item()}]")

        elapsed_time = time.time() - start_time
        logger.debug(f"[Elapsed time of all epochs: {elapsed_time}]")
            
        # Save results
        if options.save:
            result_dataloader = RandomBatchLoader(z, options.batch_size)
            result_each_time_dir = resolve_path(result_dir, f"{i}")

            all_best_images = torch.tensor([]).to(device)

            # Calc 
            for _, batch in enumerate(result_dataloader):
                images = G(batch)

                best_image, _ = get_best_image(A, images, img_size, all_target_features)

                all_best_images = torch.cat((all_best_images, best_image.unsqueeze(0)))


            # Save all best images
            best_images_path = resolve_path(result_each_time_dir, f"best_images.png")
            save_image(all_best_images, best_images_path, normalize=True, nrow=options.iterations)
            logger.info(f"[Saved all best images: {best_images_path}]")


            # Save the best image
            best_image, best_cosine_similarity = get_best_image(A, all_best_images, img_size, all_target_features)
            best_image_path = resolve_path(result_each_time_dir, f"best_image.png")
            save_image(best_image, best_image_path, normalize=True, nrow=options.iterations)
            logger.info(f"[Saved the best image: {best_image_path}] [Cosine Similarity: ({best_cosine_similarity})]")

            # Save all target images
            target_images_path = resolve_path(result_dir, f"target_images.png")
            save_image(all_target_images, target_images_path, normalize=True, nrow=options.iterations)
            logger.info(f"[Saved all target images: {target_images_path}]")

if __name__ == '__main__':
	main()