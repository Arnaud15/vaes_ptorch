"""Neural network modules used in experiments."""
from collections import namedtuple
from typing import List

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
