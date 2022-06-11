import sys

sys.path.append("../")
sys.path.append("../../")

import os
import argparse
import glob
import time
import datetime
import PIL
from typing import Any, Tuple

import torch
from torch import nn
from torch import optim
from torchvision import transforms
from torchvision.utils import save_image

from utils.util import (save_json, load_json, create_logger, resolve_path, RandomBatchLoader,
    load_model_as_feature_extractor, load_attacker_discriminator, load_attacker_generator, get_img_size)

from utils.pytorch_GAN_zoo.hubconf import StyleGAN

def get_options() -> Any:
    parser = argparse.ArgumentParser()

    # Timestamp
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    parser.add_argument("--identifier", type=str, default=time_stamp, help="timestamp")

    # Directories
    parser.add_argument("--result_dir", type=str, default="~/nas/results/common/step3", help="path to directory which includes results")
    parser.add_argument("--step2_dir", type=str, default="pure_facenet_500epoch_features", help="path to directory which includes the step2 result")
    parser.add_argument("--target_image_dir", type=str, default="~/nas/dataset/target_images", help="path to directory which contains target images")
    parser.add_argument('--target_model_path', type=str, help='path to pretrained model')

    # For inference
    parser.add_argument("--gpu_idx", type=int, default=0, help="index of cuda devices")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--epochs", type=int, default=int(1e9), help="times to initialize z") 
    parser.add_argument("--iterations", type=int, default=8, help="iterations to optimize z")
    parser.add_argument("--learning_rate", type=float, default=0.02, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="learning rate")
    parser.add_argument("--lambda_i", type=float, default=100, help="learning rate")

    # Conditions
    parser.add_argument("--embedding_size", type=int, default=512, help="dimensionality of the latent space")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--target_model", type=str, default="FaceNet", help="target model: 'FaceNet', 'Arcface', 'Magface'")
    parser.add_argument("--target_dir_name", type=str, default="Okano_MTCNN160", help="directory name which contains images of target user(features)")
    parser.add_argument("--single_mode", type=bool, default=True, help="True if you use just one stored feature(features)")

    opt = parser.parse_args()

    return opt

def L_prior(D: nn.Module, G: nn.Module, z: torch.Tensor) -> torch.Tensor:
    return torch.mean(-D(G(z)))

def calc_id_loss_for_features(G: nn.Module, T: nn.Module, z: torch.Tensor, device: str, all_target_features: torch.Tensor, image_size: int) -> torch.Tensor:
    resize = transforms.Resize((image_size, image_size))
    metric = nn.CosineSimilarity(dim=1)
    Gz_features = T(resize(G(z))).to(device)
    dim = Gz_features.shape[1]

    sum_of_cosine_similarity = 0
    for target_feature in all_target_features.view(-1, dim):
        target_feature = target_feature.expand(Gz_features.shape[0], -1)
        sum_of_cosine_similarity += metric(target_feature, Gz_features)
    
    return 1 - torch.mean(sum_of_cosine_similarity / all_target_features.shape[0])


def get_best_image(T: nn.Module, images: nn.Module, image_size: int, all_target_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    resize = transforms.Resize((image_size, image_size))
    metric = nn.CosineSimilarity(dim=1)
    Tz_features = T(resize(images))
    dim = Tz_features.shape[1]
    sum_of_cosine_similarity = 0
    for target_feature in all_target_features.view(-1, dim):
        target_feature = target_feature.expand(Tz_features.shape[0], -1)
        sum_of_cosine_similarity += metric(Tz_features, target_feature)
    sum_of_cosine_similarity /= all_target_features.shape[0]
    bestImageIndex = sum_of_cosine_similarity.argmax()
    return images[bestImageIndex], sum_of_cosine_similarity[bestImageIndex]


def main():
    options = get_options()

    # Decide device
    device = f"cuda:{options.gpu_idx}" if torch.cuda.is_available() else "cpu"
    options.device = device

    # Load step1 and step2 options
    step2_dir = "~/nas/results/common/step2/" + options.step2_dir
    step1_options = load_json(resolve_path(step2_dir, "step1.json"))
    step2_options = load_json(resolve_path(step2_dir, "step2.json"))
    label_map = load_json(resolve_path(step2_dir, "label_map.json"))

    # Create directory to save results
    result_dir = resolve_path(options.result_dir, options.identifier)
    os.makedirs(result_dir, exist_ok=True)
    

    # Save step1 and step2 options
    save_json(resolve_path(result_dir, "step1.json"), step1_options)
    save_json(resolve_path(result_dir, "step2.json"), step2_options)
    save_json(resolve_path(result_dir, "label_map.json"), label_map)

    # Save options by json format
    save_json(resolve_path(result_dir, "step3.json"), vars(options))

    # Create logger
    logger = create_logger(f"Step 3", resolve_path(result_dir, "training.log"))

    # Log options
    logger.info(vars(options))

    # Load models
    img_size = get_img_size(options.target_model)
    img_shape = (options.img_channels, img_size, img_size)
    # StyleGAN (Can't convert correctly)
    # dcgan = StyleGAN(pretrained=True)
    # D = dcgan.getNetD().to(device)
    # G = dcgan.getNetG().to(device)

    # Normal GAN
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
    T, _ = load_model_as_feature_extractor(
        arch=options.target_model,
        embedding_size=options.embedding_size,
        mode='eval',
        pretrained=True,
        path=options.target_model_path
    )
    T.to(device)
    
    D.eval()
    G.eval()
    T.eval()

    target_imagefolder_path = resolve_path(options.target_image_dir, options.target_dir_name)
    all_target_images = torch.tensor([]).to(device)
    all_target_features = torch.tensor([]).to(device)
    resize = transforms.Resize((img_size, img_size))
    convert_tensor = transforms.ToTensor()

    # Convert all taget images in target imagefolder to features
    for filename in glob.glob(target_imagefolder_path + "/*.*"):
        target_image = PIL.Image.open(filename)
        converted_target_image = convert_tensor(target_image)
        converted_target_image = resize(converted_target_image).to(device)
        all_target_images= torch.cat((all_target_images, converted_target_image.unsqueeze(0)))

        target_feature = T(converted_target_image.view(1, -1, img_size, img_size)).detach().to(device)
        all_target_features= torch.cat((all_target_features, target_feature.unsqueeze(0)))

        if options.single_mode:
            break

    # Search z^
    z = torch.randn(options.batch_size * options.iterations, options.latent_dim, requires_grad=True, device=device)
    optimizer = optim.Adam([z], lr=options.learning_rate, betas=(0.9, 0.999), weight_decay=0)
    dataloader = RandomBatchLoader(z, options.batch_size)

    # Optimize z
    start_time = time.time()
    best_total_loss_avg = 1e9
    loss_update_counter = 0
    all_best_images = torch.tensor([]).to(device)
    for epoch in range(1, options.epochs + 1):
        L_prior_loss_avg, L_id_loss_avg, total_loss_avg = 0, 0, 0

        if loss_update_counter >= 20:
            break

        for _, batch in enumerate(dataloader):
            optimizer.zero_grad()

            L_prior_loss = L_prior(D, G, batch)
            L_id_loss = calc_id_loss_for_features(G, T, batch, device, all_target_features, img_size) 
            
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

        logger.info(f"[{epoch}/{options.epochs}],\
                    [Loss: {total_loss_avg.item()},\
                    L_prior: {L_prior_loss_avg.item()},\
                    L_id: {L_id_loss_avg.item()}]")

        if epoch % 20 == 0:

            # Save results
            result_dataloader = RandomBatchLoader(z, options.batch_size)
            result_each_time_dir = resolve_path(result_dir, f"{epoch}")
            os.makedirs(result_each_time_dir, exist_ok=True)

            all_best_images = torch.tensor([]).to(device)

            # Calc 
            for _, batch in enumerate(result_dataloader):
                images = G(batch)

                # Get the best image
                best_image, _ = get_best_image(T, images, img_size, all_target_features)

                all_best_images = torch.cat((all_best_images, best_image.unsqueeze(0)))
            
            # Save all best images
            best_images_path = resolve_path(result_each_time_dir, f"best_images.png")
            save_image(all_best_images, best_images_path, normalize=True, nrow=options.batch_size * options.iterations)
            logger.info(f"[Saved all best images: {best_images_path}]")


            # Save the best image
            best_image, best_cosine_similarity = get_best_image(T, all_best_images, img_size, all_target_features)
            best_image_path = resolve_path(result_each_time_dir, f"best_image_cossim_{best_cosine_similarity}.png")

            save_image(best_image, best_image_path, normalize=True, nrow=options.iterations)
            logger.info(f"[Saved the best image: {best_image_path}] [Cosine Similarity: ({best_cosine_similarity})]")

    # Save all target images
    target_images_path = resolve_path(result_dir, f"target_images.png")
    save_image(all_target_images, target_images_path, normalize=True, nrow=options.iterations)
    logger.info(f"[Saved all target images: {target_images_path}]")
            
    elapsed_time = time.time() - start_time
    logger.debug(f"[Elapsed time of all epochs: {elapsed_time}]")


if __name__ == '__main__':
	main()