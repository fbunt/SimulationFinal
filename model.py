import torch
from torch import nn


class ThreeBodyMLP(torch.nn.Module):
    def __init__(self, n_out=10, n_hidden=10, n_hidden_size=128):
        # (t, x2x, x2y)
        n_in = 3
        # n_out:
        # full 10: (x1x, x1y, x2x, x2y, v1x, v1y, v2x, v2y, v3x, v3y)
        # pos 4: (x1x, x1y, x2x, x2y)
        # vel 6: (v1x, v1y, v2x, v2y, v3x, v3y)
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


class ThreeBodyMLPSkip(torch.nn.Module):
    def __init__(self, n_out=10, n_hidden=10, n_hidden_size=128):
        # (t, x2x, x2y)
        n_in = 3
        # n_out:
        # full 10: (x1x, x1y, x2x, x2y, v1x, v1y, v2x, v2y, v3x, v3y)
        # pos 4: (x1x, x1y, x2x, x2y)
        # vel 6: (v1x, v1y, v2x, v2y, v3x, v3y)
        super().__init__()
        self.in_ = nn.Sequential(
            nn.Linear(n_in, n_hidden_size),
            nn.BatchNorm1d(n_hidden_size),
            nn.ReLU(),
        )
        nskip = n_hidden // 2
        layers = []
        for _ in range(nskip - 1):
            layers.append(nn.Linear(n_hidden_size, n_hidden_size))
            layers.append(nn.BatchNorm1d(n_hidden_size))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(n_hidden_size, n_hidden_size))
        layers.append(nn.BatchNorm1d(n_hidden_size))
        self.block1 = nn.Sequential(*layers)
        self.block1_act = nn.ReLU(inplace=True)

        layers = []
        for _ in range(n_hidden - nskip - 1):
            layers.append(nn.Linear(n_hidden_size, n_hidden_size))
            layers.append(nn.BatchNorm1d(n_hidden_size))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(n_hidden_size, n_hidden_size))
        layers.append(nn.BatchNorm1d(n_hidden_size))
        self.block2 = nn.Sequential(*layers)
        self.block2_act = nn.ReLU(inplace=True)

        self.out = nn.Linear(n_hidden_size, n_out)

    def forward(self, x):
        x = self.in_(x)
        x = self.block1_act(self.block1(x) + x)
        x = self.block2_act(self.block2(x) + x)
        return self.out(x)
