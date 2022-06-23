import sys

sys.path.append('../')
sys.path.append('../../')
import os
import time
import datetime
import argparse
from typing import Any, Tuple

import torch
from torch import nn
from torchvision import transforms
from torchvision.utils import save_image
from torch import optim

from utils.lfw import LFW
from utils.util import (resolve_path, save_json, create_logger, get_img_size,load_attacker_discriminator, load_attacker_generator, 
                        load_json, load_model_as_feature_extractor, RandomBatchLoader, get_img_size, get_freer_gpu)

def get_options() -> Any:
    parser = argparse.ArgumentParser()

    # Timestamp
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    parser.add_argument("--identifier", type=str, default=time_stamp, help="timestamp")
    parser.add_argument("--gpu_idx", type=int, default=0, help="index of cuda devices")
    parser.add_argument("--multi_gpu", action='store_true', help="flag of multi gpu")
    
    # Dir
    parser.add_argument("--result_dir", type=str, default="../../..//results/dataset_reconstructed", help="path to directory which includes results")
    parser.add_argument("--GAN_dir", type=str, default="../../../results/common/step2/pure_facenet_500epoch_features", help="path to directory which includes the step1 result")
    parser.add_argument('--attack_model_path', default='', type=str, help='path to pretrained attack model')

    # For inference 
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--epochs", type=int, default=100, help="times to initialize z") 
    parser.add_argument("--learning_rate", type=float, default=0.035, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="learning rate")
    parser.add_argument("--lambda_i", type=float, default=100, help="learning rate")

    # Conditions
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--attack_model", type=str, default="FaceNet", help="attack model: 'FaceNet', 'Arcface', 'Magface")
    parser.add_argument("--AE_ver", type=int, default=1.1, help="AE version: 1, 1.1, 1.2, 1.3, 1.4")
    parser.add_argument("--embedding_size", type=int, default=512, help="embedding size of features of target model:[128, 512]")
    parser.add_argument("--num_of_images", type=int, default=300, help="size of test dataset")

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
    options = get_options()

    gpu_idx = get_freer_gpu()
    device = f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu"

    options.device = device

def main():
    set_global()
    # Create directory to save results

    result_dir = resolve_path(options.result_dir, (options.identifier + '_' + options.attack_model))
    os.makedirs(result_dir, exist_ok=False)
    
    # Save options by json format
    save_json(resolve_path(result_dir, "step2.json"), vars(options))

    # Create logger
    logger = create_logger(f"Step 2", resolve_path(result_dir, "inference.log"))

    GAN_options = load_json(resolve_path(options.GAN_dir, "step2.json"))

    # Log options
    logger.info(vars(options))

    # Load models
    img_size_A = get_img_size(options.attack_model)
    img_shape_A = (options.img_channels, img_size_A, img_size_A)

    D = load_attacker_discriminator(
        path=resolve_path(options.GAN_dir, "D.pth"),
        input_dim=GAN_options["img_channels"],
        network_dim=GAN_options["D_network_dim"],
        img_shape=img_shape_A,
        device=device
    ).to(device)
    G = load_attacker_generator(
        path=resolve_path(options.GAN_dir, "G.pth"),
        latent_dim=GAN_options["latent_dim"],
        network_dim=GAN_options["G_network_dim"],
        img_shape=img_shape_A,
        device=device
    ).to(device)
    A, _ = load_model_as_feature_extractor(
        arch=options.attack_model,
        embedding_size=options.embedding_size,
        mode='eval',
        path=options.attack_model_path,
        pretrained=True
    )

    if isinstance(D, nn.Module):
        D.to(device) 
        D.eval()
    if isinstance(G, nn.Module):
        G.to(device) 
        G.eval()
    if isinstance(A, nn.Module):
        A.to(device) 
        A.eval()

    if options.multi_gpu:
        D = nn.DataParallel(D)
        G = nn.DataParallel(G)
        A = nn.DataParallel(A)

    transform_A=transforms.Compose([
        transforms.Resize((img_size_A, img_size_A)),
    ])

    dataset = LFW(
        base_dir='../../../dataset/LFWA/lfw-deepfunneled-MTCNN160',
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )
    used_identity = set()
    reconstruction_count = 0

    for i, (data, label) in enumerate(dataset):
        if reconstruction_count >= options.num_of_images:
            break
        data = data.to(device)
        target_feature = A(transform_A(data).unsqueeze(0))

        folder_name = label[:label.rfind('.')]
        identity_name = folder_name[:label.rfind('_')]
        if identity_name in used_identity:
            continue
        else:
            used_identity.add(identity_name)
            reconstruction_count += 1
            
        reconstructed_result_dir = resolve_path(result_dir, folder_name)
        os.makedirs(reconstructed_result_dir, exist_ok=False)

        # Search z 
        iteration = int(256 / options.batch_size)
        z = torch.randn(options.batch_size * iteration, options.latent_dim, requires_grad=True, device=device) 
        optimizer = optim.Adam([z], lr=options.learning_rate, betas=(0.9, 0.999), weight_decay=0)
        dataloader = RandomBatchLoader(z, options.batch_size)

        # Optimize z
        start_time = time.time()
        best_total_loss_avg = 1e9
        loss_update_counter = 0
        for epoch in range(1, options.epochs + 1):
            L_prior_loss_avg, L_id_loss_avg, total_loss_avg = 0, 0, 0

            if loss_update_counter >= 20:
                break

            for _, batch in enumerate(dataloader):
                optimizer.zero_grad()

                L_prior_loss = L_prior(D, G, batch)
                L_id_loss = calc_id_loss(G, A, batch, device, target_feature, img_size_A) 
                
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

            if total_loss_avg.item() < best_total_loss_avg:
                best_total_loss_avg = total_loss_avg.item()
                loss_update_counter = 0
            else:
                loss_update_counter += 1

        logger.info(f"{label} image has been reconstructed in {epoch} epochs")
            
        # Save results
        result_dataloader = RandomBatchLoader(z, options.batch_size)

        # Calc 
        for _, batch in enumerate(result_dataloader):
            images = G(batch)

            best_image, best_cossim = get_best_image(A, images, img_size_A, target_feature)

            best_images_path = resolve_path(reconstructed_result_dir, f"best_images_{best_cossim}.png")
            save_image(best_image, best_images_path, normalize=True, nrow=iteration)

        logger.info(f"[Saved all best images: {reconstructed_result_dir}]")

        # Save all target images
        target_images_path = resolve_path(reconstructed_result_dir, f"target_images.png")
        save_image(data, target_images_path, normalize=True)

        print(f'{reconstruction_count}/{options.num_of_images} has been done')

    elapsed_time = time.time() - start_time
    logger.debug(f"[Elapsed time of all epochs: {elapsed_time}]")

if __name__=='__main__':
    main()