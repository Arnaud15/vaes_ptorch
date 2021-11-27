from collections import namedtuple
from typing import List

import torch
import torch.nn as nn
from torch import Tensor
from vaes_ptorch.utils import sample_gaussian

VaeOutput = namedtuple("VaeOutput", ["mu_x", "sig_x", "mu_z", "sig_z"])


class VAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        mu_z, sig_z = self.encoder(x)
        posterior_z = sample_gaussian(mu=mu_z, std=sig_z)
        mu_x, sig_x = self.decoder(posterior_z)
        return VaeOutput(mu_x=mu_x, sig_x=sig_x, mu_z=mu_z, sig_z=sig_z)


def mlp_layer(in_dim: int, out_dim: int):
    return nn.Sequential(nn.BatchNorm1d(in_dim), nn.Linear(in_dim, out_dim), nn.ReLU())


class MLPEncoder(nn.Module):
    def __init__(self, dims: List[int]):
        super(MLPEncoder, self).__init__()
        dims = dims + [2]  # output a mean and std param
        self.layers = nn.Sequential(
            *[mlp_layer(dims[ix - 1], dims[ix]) for ix in range(1, len(dims))]
        )

    def forward(self, x):
        out = self.layers(x)
        return out[:, 0], torch.exp(out[:, 1])
