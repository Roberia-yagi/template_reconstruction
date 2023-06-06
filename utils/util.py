import os
import sys
import re

sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append("../")
from easydict import EasyDict as edict
import json
import logging
import PIL
import glob
from termcolor import cprint
from typing import Any, Union, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models 
import torchvision.transforms as transforms
import numpy as np

import facenet_pytorch
from arcface_pytorch.model import Backbone
import magface_pytorch
from magface_pytorch.inference import network_inf
from magface_pytorch.models import magface 

from my_models.AutoEncoder import AutoEncoder
from my_models.WGAN_Discriminator import WGAN_Discriminator
from my_models.WGAN_Generator import WGAN_Generator
from my_models.stylegan2 import StyleGAN2_Generator
from my_models.stylegan2 import StyleGAN2_Discriminator

#---------------------------------------

def save_json(path: str, obj: Any):
    with open(path, "w") as json_file:
        json.dump(obj, json_file, indent=4)

def load_json(path: str) -> Any:
    with open(path) as json_obj:
        return edict(json.load(json_obj))

#---------------------------------------

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

#---------------------------------------

def resolve_path(*pathes: Tuple[str]) -> str:
    return os.path.expanduser(os.path.join(*pathes))

def remove_path_prefix(path: str) -> str:
    return path[path.rfind('/')+1:].replace(' ', '')

#---------------------------------------

# To be refactored
# def extract_target_features(T, img_size, target_image_dir, target_dir_name, single_mode, device):
#     target_imagefolder_path = resolve_path(target_image_dir, target_dir_name)
#     resize = transforms.Resize((img_size, img_size))
#     convert_tensor = transforms.ToTensor()
#     all_target_images = torch.tensor([]).to(device)
#     all_target_features = torch.tensor([]).to(device)

#     # Convert all taget images in target imagefolder to features
#     for filename in glob.glob(target_imagefolder_path + "/best_image/*.*"):
#         target_image = PIL.Image.open(filename)
#         converted_target_image = convert_tensor(target_image)
#         converted_target_image = resize(converted_target_image).to(device)
#         all_target_images= torch.cat((all_target_images, converted_target_image.unsqueeze(0)))

#         target_feature = T(converted_target_image.view(1, -1, img_size, img_size)).detach().to(device)
#         all_target_features= torch.cat((all_target_features, target_feature.unsqueeze(0)))

#         if single_mode:
#             break

#     return all_target_images, all_target_features

#---------------------------------------

def load_model_as_feature_extractor(arch: str, embedding_size: int, mode: str, path: str, pretrained=False) -> Tuple[nn.Module, int]:
    if not mode in ['train', 'eval']:
        raise("Model mode is incorrect")
    if not embedding_size in [128, 512]:
        raise("Embedding size should be 128 or 512")
    if not pretrained in [True, False]:
        raise("pretrained should be True or False")
    if pretrained and embedding_size == 128 and path is None:
        raise("128 dim pretrained model requires path")
    if mode == 'eval' and not pretrained:
        raise("Evalation model can't be used without pretrained")

    load_status = 'Not loaded'

    if arch == "FaceNet":
        model = facenet_pytorch.InceptionResnetV1(classify=False, num_classes=None, pretrained="vggface2")
        model.classify=False
        if embedding_size == 128:
            for param in model.parameters():
                param.requires_grad = False
            model.last_linear = nn.Linear(in_features=1792, out_features=128, bias=False)
            model.last_bn = nn.BatchNorm1d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
            model.logits = nn.Linear(in_features=128, out_features=44052, bias=True)
            if pretrained:
                load_status = model.load_state_dict(torch.load(path)) 
        elif embedding_size == 512:
            path = 'Pretrained by original developer'
            load_status = 'Successfully Loaded'

        if mode == 'train':
            model.train()
        elif mode == 'eval':
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

    if arch == "Arcface":
        if embedding_size == 512:
            model = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se')
            path = '../../../models/arcface_ir_se50.pth'
        elif embedding_size == 128:
            # Finetuned
            model = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se')
            for param in model.parameters():
                param.requires_grad = False
            model.output_layer[3]= torch.nn.Linear(in_features=model.output_layer[3].in_features,
                                                    out_features=128,
                                                    bias=True)
            model.output_layer[4] = torch.nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        if pretrained:
            model.load_state_dict(torch.load(path)) 
            load_status = 'Successfully Loaded'

        if mode == 'train':
            model.train()
        elif mode == 'eval':
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
        

    if arch == "Magface":
        #TODO: rewrite args according with original finetuner.py
        #TODO: understand the difference between the ways of loading model
        # if mode == 'train':
        #     model = magface.builder(args)
        #     model = magface.models.magface.load_dict_inf(args, model)
        #     for param in model.parameters():
        #         param.requires_grad = False

        #     # replace layers for small features
        #     model.features.fc = nn.Linear(in_features=model.features.fc.in_features, out_features=128, bias=True)
        #     model.features.features = nn.BatchNorm1d(128, eps=model.features.features.eps, momentum=0.9, affine=True, track_running_stats=True)
        #     model.fc = magface.MagLinear(in_features=128, out_features=args.last_fc_size)

        if mode == 'eval':
            if embedding_size == 512:
                path = '../../../models/magface_epoch_00025.pth'
                args = edict({
                    'arch':'iresnet100',
                    'cpu_mode':True,
                    'resume':path,
                    'embedding_size':embedding_size,
                    'last_fc_size':85742,
                    })
                model = network_inf.NetworkBuilder_inf(args)
                model = network_inf.load_dict_inf(args, model)
                load_status = 'Successfully Loaded'

            elif embedding_size == 128:
                args = edict({
                    'arch':'iresnet100',
                    'cpu_mode':True,
                    'resume':path,
                    'embedding_size':embedding_size,
                    'last_fc_size':85742,
                    })
                model = network_inf.NetworkBuilder_inf(args)
                model.fc = magface.MagLinear(in_features=embedding_size, out_features=args.last_fc_size)
                load_status = model.load_state_dict(torch.load(args.resume)['state_dict'])
                model.fc = nn.Identity()
            
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    if params_to_update == []:
        params_to_update = model.parameters()
    
    cprint(f'{arch} is loaded', 'green')
    cprint(f'path:{path}', 'green')
    cprint(f'train mode:{model.training}', 'green')
    cprint(f'pretrained:{load_status}', 'green')
    cprint(f'embedding size:{get_output_shape(model, get_img_size(arch))[1]}\n', 'green')

    return model, params_to_update

#---------------------------------------

def load_autoencoder(pretrained: bool, model_path: str, mode: str, ver:int):
    if not mode in ['train', 'eval']:
        raise("Model mode is incorrect")
    model = AutoEncoder(ver)
    if pretrained:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    if mode == 'train':
        for param in model.parameters():
            param.requires_grad = True
        model.train()
    elif mode == 'eval':
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        
    return model

#---------------------------------------

def get_output_shape(model, image_dim):
    return model(torch.rand(2, 3, image_dim ,image_dim)).data.shape

#---------------------------------------

def get_img_size(model: str) -> int:
    assert model in ["FaceNet", "Arcface", "Magface"]
    if model == "FaceNet":
        img_size = 160
    if model == "Arcface":
        img_size = 112
    if model == "Magface":
        img_size = 112
    
    return img_size

#---------------------------------------

def load_WGAN_discriminator(path: str, input_dim: int, network_dim: int, device) -> nn.Module:
    D = WGAN_Discriminator(input_dim=input_dim, network_dim=network_dim)
    D.load_state_dict(torch.load(path, device))

    for param in D.parameters():
        param.requires_grad = False
    
    D.eval()

    return D

def load_StyleGAN_discriminator(device):
    # size, latent, n_mlp, channel_multiplier
    D = StyleGAN2_Discriminator(
        1024
    ).to(device)
    checkpoint = torch.load('/home/akasaka/nas/models/stylegan2-ffhq-config-f.pt')
    print(checkpoint.keys())
    D.load_state_dict(checkpoint['d'])

    for param in D.parameters():
        param.requires_grad = False
    
    D.eval()

    return D

def load_WGAN_generator(path: str, network_dim:int, device) -> Tuple[nn.Module, int]:
    G = WGAN_Generator(latent_dim=100, network_dim=network_dim)
    G.load_state_dict(torch.load(path, device))

    for param in G.parameters():
        param.requires_grad = False
    
    G.eval()

    return G, 100

def load_StyleGAN_generator(truncation:int, truncation_mean:int, device) -> Tuple[nn.Module, int, int]:
    # size, latent, n_mlp, channel_multiplier
    G = StyleGAN2_Generator(
        1024, 512, 8, channel_multiplier=2
    ).to(device)
    checkpoint = torch.load('/home/akasaka/nas/models/stylegan2-ffhq-config-f.pt')
    G.load_state_dict(checkpoint["g_ema"])
    if truncation < 1:
        with torch.no_grad():
            mean_latent = G.mean_latent(truncation_mean)
    else:
        mean_latent = None

    for param in G.parameters():
        param.requires_grad = False
    
    G.eval()

    return G, mean_latent, 512


#---------------------------------------

class BatchLoader:
    def __init__(self, x: list, batch_size: int):
        self.x = x
        self.batch_size = batch_size

        self.idx = list(range(len(self.x)))
        self.current = 0

    def __iter__(self):
        self.idx = list(range(len(self.x)))
        self.current = 0
        return self

    def __next__(self):
        start = self.batch_size * self.current 
        end = self.batch_size * (self.current + 1)

        if start >= len(self.x):
            raise StopIteration()

        self.current += 1

        return self.x[start:end]

#---------------------------------------

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

#---------------------------------------

def get_memory_usage():
    memory_available = np.empty(3)
    for i in range(3):
        os.system(f'nvidia-smi -i {i} -q -d Memory |grep -A5 GPU|grep Free >tmp')
        lines = open('tmp', 'r').readlines()
        if lines == []:
            memory_available[i] = 0
        else:
            for x in lines:
                memory_available[i] = re.sub(r"\D", "", x)

    return memory_available

def get_freer_gpu():
    memory_available = np.empty(3)
    for i in range(3):
        os.system(f'nvidia-smi -i {i} --query-gpu=utilization.gpu --format=csv | grep %> tmp')
        lines = open('tmp', 'r').readlines()
        if lines == []:
            memory_available[i] = 100
        else:
            memory_available[i] = re.sub(r"\D", "", lines[1])

    freest_gpu = np.argmin(memory_available)

    if freest_gpu > 5:
        return -1

    for i in range(torch.cuda.device_count()):
        memory_available_before = get_memory_usage()
        device = f'cuda:{i}'
        tmp = torch.tensor([1000]).to(device)
        memory_available_after = get_memory_usage()
        if max(memory_available_before - memory_available_after) == 0:
            continue
        gpu_idx = np.argmax(memory_available_before - memory_available_after)
        del tmp
        torch.cuda.empty_cache()
        if gpu_idx == freest_gpu:
            return i
    
    return -1

#---------------------------------------

def align_face_image(imgs: torch.Tensor, dataset, model, detector) -> torch.Tensor:
    toPIL = transforms.ToPILImage()
    toTensor = transforms.ToTensor()
    if not model in ['FaceNet', 'Arcface', 'Magface']:
        raise('Model error in align face image')

    if imgs.dim() == 3:
        imgs = imgs.unsqueeze(0)

    if model == 'FaceNet':
        res = imgs
    elif model == 'Arcface' or 'Magface':
        res = torch.Tensor()
        # if the tensor has one image, unsqueeze it for "for loop"
        for img in imgs:
            img = toPIL(img)
            aligned_img = detector.align(img, dataset)
            if dataset == 'GAN':
                if aligned_img is None:
                    aligned_img = img
                aligned_img = toTensor(aligned_img)
                res = torch.cat((res, aligned_img.unsqueeze(0)))
            else:
                if aligned_img is not None:
                    aligned_img = toTensor(aligned_img)
                    res = torch.cat((res, aligned_img.unsqueeze(0)))

    if res.dim() != 4:
        raise('Aligned image tensor should be 4 dimentional')
    return res

#---------------------------------------

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

#------------------------------------------

class BatchLoader:
    def __init__(self, x: list, batch_size: int):
        self.x = x
        self.batch_size = batch_size

        self.idx = list(range(len(self.x)))
        self.current = 0

    def __iter__(self):
        self.idx = list(range(len(self.x)))
        self.current = 0
        return self

    def __next__(self):
        start = self.batch_size * self.current 
        end = self.batch_size * (self.current + 1)

        if start >= len(self.x):
            raise StopIteration()

        self.current += 1

        return self.x[start:end]

#---------------------------------------

def set_global(get_options):
    options = get_options()

    # Decide device
    if options.gpu_idx is None:
        gpu_idx = get_freer_gpu()
    else:
        gpu_idx = options.gpu_idx
    device = f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu"

    options.device = device

    return device, options