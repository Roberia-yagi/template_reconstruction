import torch.nn as nn

class Transferer(nn.Module):
    def __init__(self):
        super(Transferer, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

    def forward(self, x):
        output = self.linear_relu_stack(x)
        return output