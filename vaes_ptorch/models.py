"""Pytorch modules used in experiments."""
from typing import List

import torch
import torch.nn as nn


class GaussianModel(nn.Module):
    def __init__(self, model: nn.Module, out_dim: int, min_var: float = 0.0):
        super(GaussianModel, self).__init__()
        assert out_dim > 0, out_dim
        assert min_var >= 0.0, min_var
        self.min_var = min_var
        self.out_dim = out_dim
        self.base_model = model

    def forward(self, x):
        out = self.base_model(x)
        assert (
            out.size(-1) == 2 * self.out_dim
        ), f"unexpected last output dimension, expected: {self.out_dim * 2} for mean + var, found: {out.size()}"
        return out[:, : self.out_dim], torch.exp(out[:, self.out_dim :]) + self.min_var


def mlp_layer(in_dim: int, out_dim: int):
    """Build a linear layer with ReLU activation function."""
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())


def get_mlp(in_dim: int, out_dim: int, h_dims: List[int]) -> nn.Module:
    """Build a Multi Layer Perceptron (MLP) with ReLU activations from specified dimensions."""
    if not h_dims:
        return nn.Sequential(nn.Linear(in_dim, out_dim * 2))
    else:
        return nn.Sequential(
            nn.Linear(in_dim, h_dims[0]),
            *[
                mlp_layer(h_dims[ix - 1], h_dims[ix])
                for ix in range(1, len(h_dims) - 1)
            ],
            nn.Linear(h_dims[-1], out_dim),
        )


class CNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: List[int],
        kernel_sizes: List[int],
        bn: bool,
        out_dim: int,
    ):
        super(CNN, self).__init__()
        assert in_channels > 0
        assert out_dim > 0
        assert len(out_channels) == len(kernel_sizes)

        self.layers = nn.ModuleList()
        in_c = in_channels
        for (out_c, k_size) in zip(out_channels, kernel_sizes):
            self.layers.append(
                nn.Conv2d(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=k_size,
                    padding="same",
                )
            )
            if bn:
                self.layers.append(nn.BatchNorm2d(num_features=out_c))
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_c = out_c
        self.layers.append(nn.Flatten(start_dim=1, end_dim=-1))
        self.layers.append(nn.Linear(in_features=in_c, out_features=out_dim))

    def forward(self, x):
        return self.layers(x)
