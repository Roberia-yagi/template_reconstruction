import numpy as np
import torch
import torch.nn as nn

from typing import Tuple, List

# adapted from The Secret Revealer
class Generator3(nn.Module):
    def __init__(self, latent_dim: int, network_dim:int, img_shape: Tuple[int, int, int]):
        super().__init__()
	
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.network_dim = network_dim

        def deconv(in_channels: int, out_channels: int) -> List[nn.Module]:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    output_padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(),
            )
        
        self.l1 = nn.Sequential(
            nn.Linear(
                in_features=latent_dim,
                out_features=network_dim * 8 * 4 * 4,
                bias=False),
            nn.BatchNorm1d(num_features=network_dim * 8 * 4 * 4),
            nn.ReLU())
        self.l2_5 = nn.Sequential(
            deconv(network_dim * 8, network_dim * 4),
            deconv(network_dim * 4, network_dim * 2),
            deconv(network_dim * 2, network_dim),
            nn.ConvTranspose2d(
                in_channels=network_dim,
                out_channels=3,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1),
            nn.Sigmoid())

    def forward(self, z):
        img = self.l1(z)
        img = img.view(img.size(0), -1, 4, 4)
        img = self.l2_5(img)
        return img

if __name__ == '__main__':
    batch_size = 64
    latent_dim = 100 # z (in_dim in original paper)
    network_dim = 64
    img_shape=(3, 64, 64) # output_shape (out_dim in original paper)
    g = Generator3(latent_dim=latent_dim, network_dim=network_dim, img_shape=img_shape)
    z = torch.randn(batch_size, latent_dim)
    print(z.shape)

    out = g(z)
    print(out.shape)
    # print(out[0])
 