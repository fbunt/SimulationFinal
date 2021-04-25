import torch
from torch import nn


class ThreeBodyMLP(torch.nn.Module):
    def __init__(self, n_hidden=10, n_hidden_size=128):
        # (t, x2x, x2y)
        n_in = 3
        # (x1x, x1y, x2x, x2y, v1x, v1y, v2x, v2y, v3x, v3y)
        n_out = 10
        super().__init__()
        layers = [
            nn.Linear(n_in, n_hidden_size),
            nn.BatchNorm1d(n_hidden_size),
            nn.ReLU(),
        ]
        for _ in range(1, n_hidden):
            layers.append(nn.Linear(n_hidden_size, n_hidden_size))
            layers.append(nn.BatchNorm1d(n_hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(n_hidden_size, n_out))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
