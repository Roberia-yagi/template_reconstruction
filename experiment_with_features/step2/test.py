import sys
sys.path.append("../")

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch import linalg as LA
from torch.utils.data import DataLoader
from celeba import CelebA

def main():
    batch_size = 64

    dataset = CelebA(
        base_dir='~/share/dataset/CelebA',
        usage='test',
        transform=transforms.Compose([
            transforms.CenterCrop(100),
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
    )
    
    print(len(dataset))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    for i, (imgs, _) in enumerate(dataloader):
        print(i, imgs.shape)

    vgg16 = models.vgg16(num_classes=1000)
    vgg16.load_state_dict(torch.load("sample_models/VGG16.pth"))
    layers = list(vgg16.classifier.children())[:-2]
    vgg16.classifier = nn.Sequential(*layers)

    print(vgg16)

    it = iter(dataloader)
    data = next(it)
    print("Input shape", data[0].shape)

    print("Output shape", vgg16(data[0]).shape)

    # z1 = np.random.normal(0, 1, (batch_size, 100))
    # z2 = np.random.normal(0, 1, (batch_size, 100))
    z1 = torch.randn(batch_size, 100)
    z2 = torch.randn(batch_size, 100)
    print(z1.shape, z1.mean())
    print(LA.norm(z1 - z2))

    for p in vgg16.parameters():
        print(p.requires_grad)
        break

if __name__ == '__main__':
    main()