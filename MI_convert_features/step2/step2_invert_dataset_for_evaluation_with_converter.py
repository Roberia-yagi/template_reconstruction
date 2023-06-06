import sys
import time
from unittest import result

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
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch import optim

from utils.lfw import LFW
from utils.ijb import IJB
from utils.casia_web_face import CasiaWebFace
from utils.util import (resolve_path, save_json, create_logger, get_img_size, load_StyleGAN_discriminator, load_StyleGAN_generator, load_WGAN_discriminator, load_WGAN_generator,
                        load_autoencoder, load_json, load_model_as_feature_extractor, RandomBatchLoader, get_img_size, set_global)

from utils.arcface_face_cropper.mtcnn import MTCNN

from easydict import EasyDict as edict

def get_options() -> Any:
    parser = argparse.ArgumentParser()

    # Timestamp
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    parser.add_argument("--identifier", type=str, default=time_stamp, help="timestamp")
    parser.add_argument("--gpu_idx", type=int, default=None, help="index of cuda devices")
    parser.add_argument("--multi_gpu", action='store_true', help="flag of multi gpu")

    # Model
    parser.add_argument("--GAN", type=str, default='WGAN', help="an architecture of a pretrained GAN")
    
    # Dir
    parser.add_argument("--dataset", type=str, required=True, help='test dataset:[LFWA, IJB-C, CASIA]')
    parser.add_argument("--dataset_dir", type=str, required=True, help='test dataset:[LFWA, IJB-C, CASIA]')
    parser.add_argument("--result_dir", type=str, default="../../../results/dataset_reconstructed", help="path to directory which includes results")
    parser.add_argument("--step1_dir", type=str, required=True, help="path to directory which includes the step1 result")
    parser.add_argument("--GAN_dir", type=str, default="../../../results/common/step2/pure_facenet_500epoch_features", help="path to directory which includes the step1 result")

    # For inference 
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--epochs", type=int, default=100, help="times to initialize z") 
    parser.add_argument("--learning_rate", type=float, default=0.035, help="learning rate")
    parser.add_argument("--lambda_i", type=float, default=100, help="learning rate")
    parser.add_argument("--resume", type=int, default=-1, help="image of resume")

    # Conditions
    parser.add_argument("--img_channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--num_of_identities", type=int, default=300, help="size of test dataset")
    parser.add_argument("--num_per_identity", type=int, default=2, help="size of test dataset")
    parser.add_argument("--seed", type=int, default=42, help="seed for pytorch dataloader shuffle")

    opt = parser.parse_args()

    return opt


def L_prior(D: nn.Module, G: nn.Module, z: torch.Tensor) -> torch.Tensor:
    imgs, *_ = G(z)
    return torch.mean(-D(imgs))


def calc_id_loss(G: nn.Module, FE: nn.Module, detector, z: torch.Tensor, device: str, model_name: str, all_target_features: torch.Tensor, image_size: int) -> torch.Tensor:
    global options
    metric = nn.CosineSimilarity(dim=1)
    orig_imgs, *_ = G(z)
    orig_imgs = orig_imgs.detach()

    # Works well
    transform = transforms.Resize((image_size, image_size))
    Gz_features = FE(transform(orig_imgs)).to(device)

    # Bug remains -> Impossible to probagate backwards with MTCNN implemented not by Pytorch
    # transform = transforms.Resize((image_size, image_size))
    # imgs = align_face_image(transform(orig_imgs), 'GAN', model_name, detector).to(device)
    # Gz_features = FE(imgs).to(device)

    dim = Gz_features.shape[1]

    sum_of_cosine_similarity = 0
    for target_feature in all_target_features.view(-1, dim):
        target_feature = target_feature.expand(Gz_features.shape[0], -1)
        sum_of_cosine_similarity += metric(target_feature, Gz_features)
    return 1 - torch.mean(sum_of_cosine_similarity / all_target_features.shape[0])


def get_best_image(FE: nn.Module, orig_imgs: nn.Module, detector, model_name: str, all_target_features: torch.Tensor, image_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    metric = nn.CosineSimilarity(dim=1)

    # Works well
    resize = transforms.Resize((image_size, image_size))
    FEz_features = FE(resize(orig_imgs))

    # Bug remains
    # resize = transforms.Resize((image_size, image_size))
    # imgs = align_face_image(resize(orig_imgs), 'GAN', model_name, detector).to(device)
    # if (imgs.size()[0]) == 0:
    #     return None
    # FEz_features = FE(imgs)

    dim = FEz_features.shape[1]
    sum_of_cosine_similarity = 0
    for target_feature in all_target_features.view(-1, dim):
        target_feature = target_feature.expand(FEz_features.shape[0], -1)
        sum_of_cosine_similarity += metric(FEz_features, target_feature)
    sum_of_cosine_similarity /= all_target_features.shape[0]
    bestImageIndex = sum_of_cosine_similarity.argmax()
    return orig_imgs[bestImageIndex], sum_of_cosine_similarity[bestImageIndex]


def main():
    global options
    global device
    device, options = set_global(get_options)

    # Create directory to save results
    step1_dir = options.step1_dir[options.step1_dir.rfind('/'):]
    result_dir = resolve_path(options.result_dir, (options.identifier + '_' + step1_dir))
    os.makedirs(result_dir, exist_ok=True)
    
    # Create logger
    logger = create_logger(f"Step 2", resolve_path(result_dir, "inference.log"))

    step1_dir = options.step1_dir
    step1_options = edict(load_json(resolve_path(options.step1_dir, "../step1.json")))
    GAN_options = load_json(resolve_path(options.GAN_dir, "step2.json"))

    # Save options by json format
    save_json(resolve_path(result_dir, "step1.json"), step1_options)
    save_json(resolve_path(result_dir, "step2.json"), vars(options))

    # Log options
    logger.info(vars(options))

    # Load models
    img_size_T = get_img_size(step1_options.target_model)
    img_size_A = get_img_size(step1_options.attack_model)

    # Load GAN
    if options.GAN == 'StyleGAN':
        D = load_StyleGAN_discriminator(
            device=device
        ).to(device)
        G, _, latent_dim = load_StyleGAN_generator(
            truncation=1,
            truncation_mean=4096,
            device=device
        )
        G.to(device)
    elif options.GAN == 'WGAN':
        D = load_WGAN_discriminator(
            path=resolve_path(options.GAN_dir, "D.pth"),
            input_dim=GAN_options["img_channels"],
            network_dim=GAN_options["D_network_dim"],
            device=device
        ).to(device)
        G, latent_dim = load_WGAN_generator(
            path=resolve_path(options.GAN_dir, "G.pth"),
            network_dim=GAN_options["G_network_dim"],
            device=device
        )
        G.to(device)
    else:
        raise(f'GAN {options.GAN} does not exist')

    T, _ = load_model_as_feature_extractor(
        arch=step1_options.target_model,
        embedding_size=step1_options.target_embedding_size,
        mode='eval',
        path=step1_options.target_model_path,
        pretrained=True
    )
    A, _ = load_model_as_feature_extractor(
        arch=step1_options.attack_model,
        embedding_size=step1_options.attack_embedding_size,
        mode='eval',
        path=step1_options.attack_model_path,
        pretrained=True
    )
    C = load_autoencoder(
        model_path=resolve_path(step1_dir, 'AE.pth'),
        pretrained=True,
        mode='eval',
        ver=step1_options.AE_ver
    )
    detector = MTCNN(device)

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

    if options.multi_gpu:
        D = nn.DataParallel(D)
        G = nn.DataParallel(G)
        C = nn.DataParallel(C)
        A = nn.DataParallel(A)
        T = nn.DataParallel(T)

    transform_T=transforms.Compose([
        transforms.Resize((img_size_T, img_size_T)),
    ])

    # Load datasets
    if options.dataset == 'LFW':
        dataset = LFW(
            base_dir=options.dataset_dir,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]),
        )
    elif options.dataset == 'IJB-C':
        dataset = IJB(
            base_dir=options.dataset_dir,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]),
        )
    elif options.dataset == 'CASIA':
        dataset = CasiaWebFace(
                            base_dir=options.dataset_dir,
                            usage='eval',
                            num_of_identities=5120,
                            num_per_identity=20,
                            eval_num_of_identities=options.num_of_identities,
                            eval_num_per_identity=options.num_per_identity,\
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                            ]),
                        )
    else:
        raise(f'Dataset {options.dataset} does not exist')

    # double_identity: used to make sure the identity of image has 2 different images in the dataset for Type-B experiment
    # used_identity: used to make sure the reconstructed image should not be reconstructed again
    doubled_identity = set()
    used_identity = set()
    reconstruction_count = 0

    torch.manual_seed(options.seed)
    torch.cuda.manual_seed(options.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        # Optimization:
        num_workers=os.cpu_count(),
        pin_memory=True
    )

    #########################
    # Reconstruction Starts #
    #########################

    for data, (labels, filenames) in dataloader:
        if reconstruction_count >= options.num_of_identities:
            break
        data = data.to(device)
        target_feature = C(T(transform_T(data))).detach()

        # Check if the identity of the data is unique
        if options.dataset == 'CASIA':
            reconstruction_count += 1
        elif options.dataset == 'LFW':
            label = labels[0] # labels contains only an image
            if label in used_identity:
                # Continue since the identity is already reconstructed
                continue
            elif label in doubled_identity:
                # Reconstruct the image since the identity has two different images
                used_identity.add(label)
                reconstruction_count += 1
            else:
                # Continue since it is checked that the identity has at least one image
                doubled_identity.add(label)
                continue

        if options.resume > reconstruction_count:
            continue
        
        # The result directory should be as below.
        # result dir - label1 - image1
        #            - label2 - image2

        # Create the result folder for identity
        for label, filename in zip(labels, filenames):
            result_label_dir = resolve_path(result_dir, label)
            result_filename_dir = resolve_path(result_label_dir, filename)
            if not os.path.isdir(result_label_dir):
                os.makedirs(result_label_dir, exist_ok=False)
            if not os.path.isdir(result_filename_dir):
                os.makedirs(result_filename_dir, exist_ok=False)

        # Search z 
        iteration = int(256 / options.batch_size)
        z = torch.randn(options.batch_size * iteration, latent_dim, requires_grad=True, device=device) 
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
                if options.GAN == "StyleGAN":
                    batch = batch.unsqueeze(0)
                optimizer.zero_grad()

                L_prior_loss = L_prior(D, G, batch)
                L_id_loss = calc_id_loss(G, A, detector, batch, device, step1_options.attack_model, target_feature, img_size_A) 
                
                L_id_loss = options.lambda_i * L_id_loss
                total_loss = L_prior_loss + L_id_loss

                L_prior_loss_avg += L_prior_loss
                L_id_loss_avg += L_id_loss
                total_loss_avg += total_loss

                logger.info(f"[D Loss: {L_prior_loss}] [ID Loss: {L_id_loss}], [Total Loss: {total_loss}]")

                total_loss.backward()
                optimizer.step()

                # Calculate loss average
                L_prior_loss_avg /= z.shape[0]
                L_id_loss_avg /= z.shape[0]
                total_loss_avg /= z.shape[0]

            if total_loss_avg.item() < best_total_loss_avg:
                best_total_loss_avg = total_loss_avg.item()
                loss_update_counter = 0
            else:
                loss_update_counter += 1

        for label in labels:
            logger.info(f"{label} image has been reconstructed in {epoch} epochs")

        del L_prior_loss
        del L_id_loss
        del total_loss
        del L_prior_loss_avg
        del L_id_loss_avg
        del total_loss_avg
        torch.cuda.empty_cache()

        # Calc 
        for _, batch in enumerate(z):
            batch = batch.unsqueeze(0)
            if options.GAN == "StyleGAN":
                batch = batch.unsqueeze(0)

            images = G(batch)
            images, *_ = images
            images = images.detach()

            result  = get_best_image(A, images, detector, step1_options.attack_model, target_feature, img_size_A)
            if result is not None:
                best_image, best_cossim = result
                best_images_path = resolve_path(result_filename_dir, f"best_images_{best_cossim}.png")
                save_image(best_image, best_images_path, normalize=True, nrow=iteration)

        logger.info(f"[Saved all best images: {result_filename_dir}]")

        # Save all target images
        target_images_path = resolve_path(result_filename_dir, f"target_images.png")
        save_image(data, target_images_path, normalize=True)

        logger.info(f'{reconstruction_count}/{options.num_of_identities} has been done')

    elapsed_time = time.time() - start_time
    logger.debug(f"[Elapsed time of all epochs: {elapsed_time}]")

if __name__=='__main__':
    main()