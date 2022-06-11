import os
import sys

sys.path.append("../")
import json
import logging
from typing import Any, Union, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models 
import numpy as np

import facenet_pytorch
import arcface_pytorch

from models.Discriminator3 import Discriminator3
from models.Generator3 import Generator3

def save_json(path: str, obj: Any):
    with open(path, "w") as json_file:
        json.dump(obj, json_file, indent=4)

def load_json(path: str) -> Any:
    with open(path) as json_obj:
        return json.load(json_obj)

def create_logger(name: str, path: Optional[str] = None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        "%Y-%m-%dT%H:%M:%S"
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if path is not None:
        fh = logging.FileHandler(path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger

def resolve_path(*pathes: Tuple[str]) -> str:
    return os.path.expanduser(os.path.join(*pathes))

def load_model_as_feature_extractor(model: str) -> Tuple[nn.Module, int]:
    if model == "FaceNet":
        model = facenet_pytorch.InceptionResnetV1(classify=False, num_classes=None, pretrained="vggface2")

    if model == "Arcface":
        path = "/home/akasaka/nas/models/arcface_ir_se50.pth"
        model = arcface_pytorch.model.Backbone(num_layers=50, drop_ratio=0, mode='ir_se')
        model.load_state_dict(torch.load(path)) 


    for param in model.parameters():
        param.requires_grad = False

    return model

def get_img_size(model: str) -> int:
    if model == "FaceNet":
        img_size = 160
    if model == "Arcface":
        img_size = 112
    return img_size

def load_attacker_discriminator(path: str, input_dim: int, network_dim: int, img_shape: Tuple[int, int, int]) -> nn.Module:
    D = Discriminator3(input_dim=input_dim, network_dim=network_dim, img_shape=img_shape)
    D.load_state_dict(torch.load(path))

    for param in D.parameters():
        param.requires_grad = False
    
    D.eval()

    return D

def load_attacker_generator(path: str, latent_dim: int, network_dim:int, img_shape: Tuple[int, int, int]) -> nn.Module:
    G = Generator3(latent_dim=latent_dim, network_dim=network_dim ,img_shape=img_shape)
    G.load_state_dict(torch.load(path))

    for param in G.parameters():
        param.requires_grad = False
    
    G.eval()

    return G

class RandomBatchLoader:
    def __init__(self, x: torch.Tensor, batch_size: int):
        self.x = x
        self.batch_size = batch_size

        self.idx = torch.randperm(self.x.shape[0])
        self.current = 0

    def __iter__(self):
        self.idx = torch.randperm(self.x.shape[0])
        self.current = 0
        return self

    def __next__(self):
        start = self.batch_size * self.current 
        end = self.batch_size * (self.current + 1)

        if start >= self.x.shape[0]:
            raise StopIteration()

        self.current += 1

        mask = self.idx[start:end]

        return self.x[mask]

def extract_features_from_nnModule(
                     batch: Any,
                     model: nn.Module,
                     layer_name: str,
                     device: Any):

    features = model(batch.to(device))
    if type(features) == dict:
        features = features[layer_name]
    features.detach().to(device)

    return features

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)