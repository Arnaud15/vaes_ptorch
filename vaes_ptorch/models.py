"""Pytorch modules used in experiments."""
from typing import List

import torch
import torch.nn as nn

NormalParams = namedtuple("NormalParams", ["mu", "var",])


class GaussianModel(nn.Module):
    def __init__(
        self, model: nn.Module, out_dim: int, split_dim: int = 1, min_var: float = 0.0,
    ):
        super(GaussianModel, self).__init__()
        assert out_dim > 0, out_dim
        assert min_var >= 0.0, min_var
        assert split_dim >= 0
        self.min_var = min_var
        self.out_dim = out_dim
        self.base_model = model
        self.split_dim = split_dim

    def forward(self, x):
        out = self.base_model(x)
        assert (
            out.size(self.split_dim) == 2 * self.out_dim
        ), f"unexpected last output dimension, expected: {self.out_dim * 2} for mean + var, found: {out.size()}"
        mean, unnormalized_var = torch.split(
            out, split_size_or_sections=self.out_dim, dim=self.split_dim
        )
        return NormalParams(mu=mean, var=torch.exp(unnormalized_var) + self.min_var)


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
        downsampling: List[bool],
        bn: bool,
        f_map_size: int,
        out_dim: int,
    ):
        super(CNN, self).__init__()
        assert in_channels > 0
        assert out_dim > 0
        assert f_map_size > 0
        assert len(out_channels) == len(kernel_sizes) and len(out_channels) == len(
            downsampling
        )

        self.layers = nn.ModuleList()
        in_c = in_channels
        for (out_c, k_size, downsample) in zip(
            out_channels, kernel_sizes, downsampling
        ):
            self.layers.append(
                nn.Conv2d(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=k_size,
                    padding=(k_size - 1) // 2,
                )
            )
            if bn:
                self.layers.append(nn.BatchNorm2d(num_features=out_c))
            if downsample:
                self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_c = out_c
        self.layers.append(nn.Flatten(start_dim=1, end_dim=-1))
        self.layers.append(
            nn.Linear(
                in_features=f_map_size * f_map_size * out_c, out_features=out_dim,
            )
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DeCNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        f_map_size: int,
        channel_size: int,
        out_channels: List[int],
        kernel_sizes: List[int],
        downsampling: List[bool],
        bn: bool,
    ):
        super(DeCNN, self).__init__()
        assert in_dim > 0
        assert f_map_size > 0
        assert channel_size > 0
        assert len(out_channels) == len(kernel_sizes) and len(out_channels) == len(
            downsampling
        )

        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Linear(
                in_features=in_dim, out_features=f_map_size * f_map_size * channel_size,
            )
        )
        self.layers.append(
            nn.Unflatten(
                dim=-1, unflattened_size=(channel_size, f_map_size, f_map_size)
            )
        )
        in_c = channel_size
        for (out_c, k_size, downsample) in zip(
            out_channels, kernel_sizes, downsampling
        ):
            self.layers.append(
                nn.Conv2d(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=k_size,
                    padding=(k_size - 1) // 2,
                )
            )
            if bn:
                self.layers.append(nn.BatchNorm2d(num_features=out_c))
            if downsample:
                self.layers.append(
                    nn.ConvTranspose2d(
                        in_channels=out_c, out_channels=out_c, kernel_size=2, stride=2,
                    )
                )
            in_c = out_c

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
