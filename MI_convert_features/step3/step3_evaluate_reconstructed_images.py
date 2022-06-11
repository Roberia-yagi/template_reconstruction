import sys

sys.path.append('../')
sys.path.append('../../')
import os
import time
import glob
import datetime
import argparse
from typing import Any, Tuple

import torch
from torch import nn
from torchvision import transforms
from torchvision.utils import save_image

from utils.util import (resolve_path, save_json, create_logger, get_img_size, load_json, 
                        load_model_as_feature_extractor, get_img_size, get_freer_gpu,
                        extract_target_features)

def get_options() -> Any:
    parser = argparse.ArgumentParser()

    # Timestamp
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    parser.add_argument("--identifier", type=str, default=time_stamp, help="timestamp")
    parser.add_argument("--result_dir", type=str, default="../../../results/evaluation_results", help="path to directory which includes results")
    parser.add_argument("--dataset_dir", type=str, default="../../../dataset/LFWA/lfw-deepfunneled-MTCNN160", help="path to dataset directory")
    parser.add_argument("--step2_dir", type=str, required=True, help="path to directory which includes the step2 result")
    parser.add_argument("--embedding_size", type=int, default=512, help="dimensionality of the latent space")
    parser.add_argument('--target_model_path', type=str, help='path to pretrained model')

    opt = parser.parse_args()
    return opt

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

    step2_dir = options.step2_dir[options.step2_dir.rfind('/'):][18:]
    result_dir = resolve_path(options.result_dir, (options.identifier + '_' + step2_dir))
    os.makedirs(result_dir, exist_ok=True)

    # Save options by json format
    save_json(resolve_path(result_dir, "step3.json"), vars(options))

    # Create logger
    logger = create_logger(f"Step 3", resolve_path(result_dir, "reconstruction.log"))

    step2_dir = options.step2_dir
    step2_options = load_json(resolve_path(step2_dir, "step2.json"))

    # Log options
    logger.info(vars(options))

    # Load models
    img_size_T = get_img_size(step2_options['target_model'])

    T, _ = load_model_as_feature_extractor(
        arch=step2_options['target_model'],
        embedding_size=options.embedding_size,
        mode='eval',
        path=options.target_model_path,
        pretrained=True
    )

    if isinstance(T, nn.Module):
        T.to(device) 
        T.eval()

    transform_T=transforms.Compose([
        transforms.Resize((img_size_T, img_size_T)),
    ]) 

    for folder_path in glob.glob(options.step2_dir + '/*/'):
        folder_name = folder_path[folder_path[:-1].rfind('/')+1:-1]
        _, target_feature = extract_target_features(T, img_size_T,
            options.dataset_dir, folder_name, True, device)
        _, reconstructed_features = extract_target_features(T, img_size_T,
            options.step2_dir, folder_name, True, device)
        print(target_feature)
        break
        # print(features)

if __name__ == '__main__':
	main()