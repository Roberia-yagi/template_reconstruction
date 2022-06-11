import os
import sys
import json
import logging
from typing import Any, Union, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models

from facenet_pytorch import InceptionResnetV1

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

# Get VGG16
# Model's output: last layer's output
def load_vgg16(path: str, num_classes: int) -> nn.Module:
    vgg16 = models.vgg16(num_classes=num_classes)
    vgg16.load_state_dict(torch.load(path)) 

    for param in vgg16.parameters():
        param.requires_grad = False

    vgg16.eval()

    return vgg16

# Get VGG16 modified as feature extractor
# Model's output: feature vector (4096)
def load_vgg16_as_feature_extractor(path: str, num_classes: int) -> nn.Module:
    vgg16 = models.vgg16(num_classes=num_classes)
    vgg16.load_state_dict(torch.load(path)) 

    layers = list(vgg16.classifier.children())[:-2]
    vgg16.classifier = nn.Sequential(*layers)

    for param in vgg16.parameters():
        param.requires_grad = False

    vgg16.eval()

    return vgg16

# Get FaceNet
# Model's output: last layer's output
def load_facenet(path: str, classifier_mode: str, num_classes: int) -> nn.Module:
    if classifier_mode == "multi_class":
        facenet = InceptionResnetV1(classify=True, num_classes=num_classes)
        facenet.load_state_dict(torch.load(path)) 
    elif classifier_mode == "features":
        facenet = InceptionResnetV1(classify=False, num_classes=None, pretrained="vggface2")

    for param in facenet.parameters():
        param.requires_grad = False

    facenet.eval()

    return facenet

# Get FaceNet modified as feature extractor
# Model's output: feature vector (512)
def load_facenet_as_feature_extractor(path: str, num_classes: int) -> nn.Module:
    facenet = InceptionResnetV1(classify=True, num_classes=num_classes)
    facenet.load_state_dict(torch.load(path)) 

    facenet.logits = nn.Identity()

    for param in facenet.parameters():
        param.requires_grad = False

    facenet.eval()

    return facenet

def load_classifier(model: str, classifier_mode: str, path: str, num_classes: int) -> Union[nn.Module, None]:
    if model == 'VGG16':
        return load_vgg16(path, num_classes)
    
    if model == 'FaceNet':
        return load_facenet(path, classifier_mode, num_classes)

    return None

def load_classifier_as_feature_extractor(model: str, path: str, num_classes: int) -> Union[nn.Module, None]:
    if model == 'VGG16':
        return load_vgg16_as_feature_extractor(path, num_classes)
    
    if model == 'FaceNet':
        return load_facenet_as_feature_extractor(path, num_classes)

    return None

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

def get_img_size(model: str) -> int:
    if model == "FaceNet":
        img_size = 160
    if model == "Arcface":
        img_size = 112
    return img_size