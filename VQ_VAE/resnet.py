import torch
from torch import nn


class ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, num_hiddens: int, num_residual_hiddens: int):
        super().__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_residual_hiddens,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=num_residual_hiddens,
                out_channels=num_hiddens,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(
        self,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
    ):
        super().__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList(
            [
                ResidualBlock(num_hiddens, num_hiddens, num_residual_hiddens)
                for _ in range(num_residual_layers)
            ]
        )

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)

        return torch.relu(x)
