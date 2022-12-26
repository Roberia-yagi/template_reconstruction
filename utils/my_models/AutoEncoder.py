import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, ver:int):
        super(AutoEncoder, self).__init__()
        if not ver in [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.81, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6]:
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
        elif ver == 1.5:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
            )
        elif ver == 1.6:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
            )
        elif ver == 1.7:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
            )
        elif ver == 1.8:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
            )
        elif ver == 1.81:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 512),
            )
        elif ver == 1.9:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Linear(512, 512),
            )
        elif ver == 2.0:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 512),
            )
        elif ver == 2.1:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 512),
            )
        elif ver == 2.2:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 512),
            )
        elif ver == 2.3:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
            )
        if ver == 2.4:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
            )
        if ver == 2.5:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 512),
            )
        if ver == 2.6:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(512, 512),
            )

    def forward(self, x):
        output = self.linear_relu_stack(x)
        return output