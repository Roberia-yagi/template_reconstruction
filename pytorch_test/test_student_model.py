import sys

sys.path.append("../")

import os
import onnx
import argparse
import copy
import numpy as np
import time
import datetime
import PIL
from logging import Logger
from typing import Any, List, Tuple, Dict

import onnxruntime as ort

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision import transforms
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split

from util import (save_json, load_json, create_logger, resolve_path, load_trained_resnet50)

from celeba import CelebA

def get_options() -> Any:
    parser = argparse.ArgumentParser()

    # Timestamp
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    # Path

    # Directories
    parser.add_argument("--step1_dir", type=str, default="knowledge_distillation", help="path to directory which includes the knowledge distillation result")
    parser.add_argument("--target_image_dir", type=str, default="/home/akasaka/nas/dataset/target_images", help="path to target images_dir")
    parser.add_argument("--target_model_dir", type=str, default="/home/akasaka/nas/models/", help="path to target model")

    # Conditions
    parser.add_argument("--gpu_idx", type=int, default=0, help="index of cuda devices")
    parser.add_argument("--target_image1", type=str, default="Okano_MTCNN160/001.jpg", help="path to target images")
    parser.add_argument("--target_model", type=str, default="arcface-resnet100_MS1MV3.onnx", help="path to target model")
    parser.add_argument("--img_size", type=int, default=112, help="size of image")

    opt = parser.parse_args()

    return opt


def main():
    # Get options
    options = get_options()


    # Decide device
    device = f"cuda:{options.gpu_idx}" if torch.cuda.is_available() else "cpu"
    options.device = device

    # Load previous learning options
    step1_dir = "~/nas/results/DiBiGAN/step1/" + options.step1_dir
    step1_options = load_json(resolve_path(step1_dir, "step1.json"))

    # Load target model; ONNX format
    T = onnx.load(resolve_path(options.target_model_dir, options.target_model))
    # Check that the model is well formed
    onnx.checker.check_model(T)
    # Craete inference session for Target model
    T_session = ort.InferenceSession(resolve_path(options.target_model_dir, options.target_model))

    # Load Student model; Resnet50
    S = load_trained_resnet50(
        path=resolve_path(step1_dir, f"{step1_options['identifier']}.pth"),
    ).to(device)

    # Create loss function
    criterion = nn.CosineSimilarity()

    # Load a target image
    target_image = PIL.Image.open(resolve_path(options.target_image_dir, options.target_image))
    resize = transforms.Resize((options.img_size, options.img_size))
    convert_tensor = transforms.ToTensor()
    converted_target_image = convert_tensor(target_image)
    converted_target_image = resize(converted_target_image).unsqueeze(0)

    # Inference phase
    target_out = T_session.run([], {"data":converted_target_image.numpy()})
    target_out = torch.tensor(np.array(target_out)).to(device)
    student_out = S(converted_target_image.to(device))
    cosine_similarity = torch.mean(criterion(target_out, student_out))


    print(f"cosine similarity between {options.target_model}'s output and {step1_options['identifier']}'s output for {options.target_image} is {cosine_similarity}")

if __name__ == '__main__':
	main()