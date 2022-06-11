import numpy as np
import torch
import torch.nn as nn

from typing import Tuple, List

# adapted from The Secret Revealer
class Discriminator3(nn.Module):
    def __init__(self, input_dim: int, network_dim: int, img_shape: Tuple[int, int, int]):
        super().__init__()

        def conv(in_channels: int, out_channels: int) -> List[nn.Module]:
            return nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, 
                    out_channels=out_channels,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    bias=False
                ),
                nn.InstanceNorm2d(num_features=out_channels, affine=True),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.ls = nn.Sequential(
            # Image (3x64x64)
            nn.Conv2d(
                in_channels=input_dim,
                out_channels=network_dim,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
            conv(network_dim, network_dim * 2),
            conv(network_dim * 2, network_dim * 4),
            conv(network_dim * 4, network_dim * 8),

            # State (512x4x4)
            nn.Conv2d(
                in_channels=network_dim * 8,
                out_channels=1,
                kernel_size=4,
                # stride=1,
                # padding=0,
                bias=False
            ),
        )

    def forward(self, img):
        validity = self.ls(img)
        validity = validity.view(-1)
        return validity

if __name__ == '__main__':
    batch_size = 64
    img_shape = (3, 64, 64)
    d = Discriminator3(input_dim=3, network_dim=64, img_shape=img_shape)
    img = torch.randn(batch_size, *img_shape)
    print(img.shape)

    out = d(img)
    print(out.shape)
    print(out)
    print(torch.mean(out))
    # print(out[0])
    