"""Neural network modules used in experiments."""
import torch
import torch.nn as nn

import vaes_ptorch.args as args
import vaes_ptorch.proba as proba


class GaussianNN(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        out_dim: int,
        split_dim: int = 1,
        min_var: float = args.MIN_VAR,
    ):
        """
        Neural network module to model independent multivariate gaussian distributions.

        Parameterized by a neural network module, a target dimension size for
        the multivariate gaussian and a dimension to split at.

        Wraps around the input neural network module as follows:
        - split at the given dimension the input of the neural network module
          in two components
        - the first component will be used unchanged to model the means of
          independent gaussians
        - the second component will be exponentiated to model the variance of
          the above independent gaussians.
        """
        super(GaussianNN, self).__init__()
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
        return proba.NormalParams(
            mu=mean, var=torch.exp(unnormalized_var) + self.min_var
        )


def get_mlp(in_dim: int, out_dim: int, h_dim: int, n_hidden: int) -> nn.Module:
    """Build a Multi Layer Perceptron (MLP) with residual connections, ReLU
    activations, Layer Normalization and a fixed hidden size"""
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
    """Fully Connected residual block with Layer Norm and ReLU activation"""

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
