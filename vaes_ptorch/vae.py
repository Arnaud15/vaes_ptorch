from collections import namedtuple
from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from .utils import sample_gaussian

VaeOutput = namedtuple("VaeOutput", ["mu_x", "sig_x", "mu_z", "sig_z"])


class VAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        mu_z, sig_z = self.encoder(x)
        posterior_z = sample_gaussian(mu=mu_z, var=sig_z)
        mu_x, sig_x = self.decoder(posterior_z)
        return VaeOutput(mu_x=mu_x, sig_x=sig_x, mu_z=mu_z, sig_z=sig_z)


def mlp_layer(in_dim: int, out_dim: int):
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())


class GaussianMLP(nn.Module):
    def __init__(self, in_dim: int, h_dims: List[int], out_dim: int):
        super(GaussianMLP, self).__init__()
        self.out_dim = out_dim
        if not h_dims:
            self.layers = nn.Sequential(nn.Linear(in_dim, self.out_dim * 2))
        else:
            self.layers = nn.Sequential(
                nn.Linear(in_dim, h_dims[0]),
                *[
                    mlp_layer(h_dims[ix - 1], h_dims[ix])
                    for ix in range(1, len(h_dims) - 1)
                ],
                nn.Linear(h_dims[-1], self.out_dim * 2),
            )

    def forward(self, x):
        out = self.layers(x)
        return out[:, : self.out_dim], torch.exp(out[:, self.out_dim :])
