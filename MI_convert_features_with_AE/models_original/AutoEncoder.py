import torch
from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self, input_shape: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=input_shape//2),
            nn.ReLU(),
            nn.Linear(in_features=input_shape//2, out_features=input_shape//4),
            nn.ReLU(),
            nn.Linear(in_features=input_shape//4, out_features=input_shape//2),
            nn.ReLU(),
            nn.Linear(in_features=input_shape//2, out_features=input_shape),
            nn.Sigmoid(),
        )

    def forward(self, features):
        reconstructed = self.model(features)
        return reconstructed

def main():
    AE = AutoEncoder(512)

if __name__ == '__main__':
    main()

