import os
import sys
import json
import logging
import re
from typing import Any, Union, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

import facenet_pytorch 
import arcface_pytorch

import onnx
import onnxruntime as ort

import PIL
import glob
import numpy as np

def save_json(path: str, obj: Any):
    with open(path, "w") as json_file:
        json.dump(obj, json_file, indent=4)

def load_json(path: str) -> Any:
    with open(path) as json_obj:
        return json.load(json_obj)

def parse_string_for_set(set: str) -> set:
    print(re.split('[^\(\{\,]*', str))

def load_model_as_feature_extraction(model: str) -> Tuple[nn.Module]:
    if model == "FaceNet":
        model = facenet_pytorch.InceptionResnetV1(classify=False, num_classes=None, pretrained="vggface2")
        for param in model.parameters():
            param.requires_grad = False

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


def load_onnx_model(path: str) -> ort.InferenceSession:
    # Load target model; ONNX format
    model_path = "/home/akasaka/nas/models/arcface-resnet100_MS1MV3.onnx"
    T = onnx.load(model_path)
    # Check that the model is well formed
    onnx.checker.check_model(T)
    # Craete inference session for Target model
    T_session = ort.InferenceSession(model_path)

    return T_session



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

def extract_features_from_orx(
                     batch: Any,
                     model: ort.InferenceSession,
                     device: Any):

    # Inference phase
    features = model.run([], {"data":batch.numpy()})
    features = torch.tensor(np.array(features)).squeeze(0).to(device)

    return features


# Batch size is tentitively the same as the number of images
def create_batch_from_images(imagefolder_path: str,
                             transform: transforms):
                        
    batch = torch.Tensor()
    for filename in glob.glob(imagefolder_path + "/*.jpg"):
        image = PIL.Image.open(filename)
        converted_image = transform(image)
        batch = torch.cat((batch, converted_image.unsqueeze(0)), 0)

    return batch


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

if __name__ == '__main__':
    print(arcface_pytorch)