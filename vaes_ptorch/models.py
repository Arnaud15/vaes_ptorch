"""Neural network modules used in experiments."""
import torch
import torch.distributions as tdist
import torch.nn as nn

import vaes_ptorch.args as args
import vaes_ptorch.proba as proba


def get_mlp(in_dim: int, out_dim: int, h_dim: int, n_hidden: int) -> nn.Module:
    """Build a Multi Layer Perceptron (MLP) with residual connections, ReLU
    activations, Layer Normalization and a fixed hidden size."""
    assert n_hidden >= 0, n_hidden
    if not n_hidden:
        return nn.Linear(in_dim, out_dim)
    else:
        return nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(),
            *[ResBlock(dim=h_dim) for _ in range(n_hidden)],
            nn.LayerNorm(h_dim),
            nn.Linear(h_dim, out_dim),
        )


class ResBlock(nn.Module):
    """Fully Connected residual block with Layer Norm and ReLU activation."""

    def __init__(self, dim: int):
        super(ResBlock, self).__init__()
        assert dim > 0, dim
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.lin = nn.Linear(dim, dim)

    def forward(self, x):
        assert x.size(-1) == self.dim, (x.size(), self.dim)
        res = x
        x = self.norm(x)
        x = self.lin(x)
        x = nn.ReLU()(x)
        return x + res
