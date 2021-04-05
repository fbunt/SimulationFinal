import torch
from torch import nn


class ThreeBodyMLP(torch.nn.module):
    def __init__(self, n_in=3, n_hidden=10, n_hidden_size=128, n_out=10):
        super().__init__()
        layers = [nn.Linear(n_in, n_hidden), nn.GELU(inplace=True)]
        for _ in range(1, n_hidden_size):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.GELU(inplace=True))
        layers.append(nn.Linear(n_hidden, n_out))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
