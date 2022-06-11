import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, ver:int):
        super(AutoEncoder, self).__init__()
        if not ver in [1, 1.1, 1.2, 1.3, 1.4]:
            raise("AutoEncoder version is illegal")
        if ver == 1:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
            )
        elif ver == 1.1:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
            )
        elif ver == 1.2:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
            )
        elif ver == 1.3:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Linear(512, 512),
            )
        elif ver == 1.4:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
            )

    def forward(self, x):
        output = self.linear_relu_stack(x)
        return output