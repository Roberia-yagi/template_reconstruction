# Invert feature to image with converter trained in step 1
import os
import sys
sys.path.append("../")
import argparse
import time
import datetime
import PIL
from typing import Any, Tuple

import torch
from torch import nn
from torch import optim
from torchvision import transforms
from torchvision.utils import save_image
from utils.util import (save_json, load_json, create_logger, resolve_path, load_model_as_feature_extractor,
    load_feature_transferer, RandomBatchLoader, BatchLoader, load_attacker_discriminator,
    load_attacker_generator, get_freer_gpu, get_img_size, extract_features_from_nnModule)

def get_options() -> Any:
    parser = argparse.ArgumentParser()

    # Timestamp
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    parser.add_argument("--identifier", type=str, default=time_stamp, help="timestamp")

    # Save
    parser.add_argument("--save", type=bool, default=True, help="if save log and data")

    # Directories
    parser.add_argument("--result_dir", type=str, default="~/nas/results/MI_convert_features/step2", help="path to directory which includes results")
    parser.add_argument("--step1_dir", type=str, default="2022_04_24_16_21", help="path to directory which includes the step1 result")
    parser.add_argument("--GAN_dir", type=str, default="~/nas/results/common/step2/pure_facenet_500epoch_features", help="path to directory which includes the step1 result")
    parser.add_argument("--target_image_dir", type=str, default="~/nas/dataset/fairface/train", help="path to directory which contains target images")

    # For inference
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--times", type=int, default=5, help="times to initialize z")
    parser.add_argument("--epochs", type=int, default=1500, help="times to initialize z") 
    parser.add_argument("--iterations", type=int, default=8, help="iterations to optimize z")
    parser.add_argument("--learning_rate", type=float, default=0.02, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="learning rate")
    parser.add_argument("--lambda_i", type=float, default=150, help="learning rate")

    # Conditions
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--target_model", type=str, default="FaceNet", help="target model: 'FaceNet', 'Arcface'")
    parser.add_argument("--attack_model", type=str, default="Arcface", help="attack model: 'FaceNet', 'Arcface'")
    parser.add_argument("--target_image_name", type=str, default="1.jpg", help="image name of target user")

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


def set_global():
    global options
    global device
    global metric
    options = get_options()
    metric = nn.CosineSimilarity(dim=1)
    # Decide device
    gpu_id = get_freer_gpu()
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    options.device = device

def main():
    set_global()
    if options.save:
        # Create directory to save results
        result_dir = resolve_path(options.result_dir, options.identifier)
        os.makedirs(result_dir, exist_ok=True)

        # Create directory to save each result
        for i in range(1, options.times + 1):
            result_each_time_dir = resolve_path(result_dir, f"{i}")
            os.makedirs(result_each_time_dir, exist_ok=True)
        
        # Save options by json format
        save_json(resolve_path(result_dir, "step2.json"), vars(options))

    # Load step1 options
    step1_dir = "~/nas/results/MI_convert_features/step1/" + options.step1_dir
    step1_options = load_json(resolve_path(step1_dir, "step1.json"))
    GAN_options = load_json(resolve_path(options.GAN_dir, "step2.json"))

    # Create logger
    if options.save:
        logger = create_logger(f"Step 2", resolve_path(result_dir, "inference.log"))
    else:
        logger = create_logger(f"Step 2")

    # Log options
    logger.info(vars(options))

    # Load models
    img_size_T = get_img_size(options.target_model)
    img_size_A = get_img_size(options.attack_model)
    img_shape_T = (options.img_channels, img_size_T, img_size_T)

    D = load_attacker_discriminator(
        path=resolve_path(options.GAN_dir, "D.pth"),
        input_dim=GAN_options["img_channels"],
        network_dim=GAN_options["D_network_dim"],
        img_shape=img_shape_T
    ).to(device)
    G = load_attacker_generator(
        path=resolve_path(options.GAN_dir, "G.pth"),
        latent_dim=options.latent_dim,
        network_dim=GAN_options["G_network_dim"],
        img_shape=img_shape_T
    ).to(device)
    T = load_model_as_feature_extractor(
        model=options.target_model
    ).to(device)
    A = load_model_as_feature_extractor(
        model=options.attack_model
    ).to(device)
    C = load_feature_transferer(
        model_path=resolve_path(step1_dir, 'C.pth')
    ).to(device)

    if isinstance(D, nn.Module):
        D.to(device) 
        D.eval()
    if isinstance(G, nn.Module):
        G.to(device) 
        G.eval()
    if isinstance(C, nn.Module):
        C.to(device) 
        C.eval()
    if isinstance(T, nn.Module):
        T.to(device) 
        T.eval()
    if isinstance(A, nn.Module):
        A.to(device) 
        A.eval()

    target_imagefolder_path = resolve_path(options.target_image_dir, options.target_image_name)
    all_target_images = torch.tensor([]).to(device)
    all_target_features_in_A = torch.tensor([]).to(device)
    transform=transforms.Compose([
        transforms.Resize((img_size_T, img_size_T)),
        transforms.ToTensor(),
    ])


    target_image = PIL.Image.open(target_imagefolder_path)
    converted_target_image = transform(target_image).to(device)
    all_target_images= torch.cat((all_target_images, converted_target_image.unsqueeze(0)))

    target_feature_in_T = T(converted_target_image.view(1, -1, img_size_T, img_size_T)).detach().to(device)
    target_feature_in_A = C(target_feature_in_T)
    all_target_features_in_A = torch.cat((all_target_features_in_A, target_feature_in_A.unsqueeze(0)))

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
                L_id_loss = calc_id_loss(G, A, batch, device, all_target_features_in_A, img_size_A) 
                
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

                best_image, _ = get_best_image(A, images, img_size_A, all_target_features_in_A)

                all_best_images = torch.cat((all_best_images, best_image.unsqueeze(0)))


            # Save all best images
            best_images_path = resolve_path(result_each_time_dir, f"best_images.png")
            save_image(all_best_images, best_images_path, normalize=True, nrow=options.iterations)
            logger.info(f"[Saved all best images: {best_images_path}]")


            # Save the best image
            best_image, best_cosine_similarity = get_best_image(A, all_best_images, img_size_A, all_target_features_in_A)
            best_image_path = resolve_path(result_each_time_dir, f"best_image.png")
            save_image(best_image, best_image_path, normalize=True, nrow=options.iterations)
            logger.info(f"[Saved the best image: {best_image_path}] [Cosine Similarity: ({best_cosine_similarity})]")

            # Save all target images
            target_images_path = resolve_path(result_dir, f"target_images.png")
            save_image(all_target_images, target_images_path, normalize=True, nrow=options.iterations)
            logger.info(f"[Saved all target images: {target_images_path}]")
    

if __name__ == '__main__':
	main()