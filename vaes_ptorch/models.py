"""Neural network modules used in experiments."""
from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor
import einops as ei


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

    def forward(self, x: Tensor) -> Tensor:
        assert x.size(-1) == self.dim, (x.size(), self.dim)
        res = x
        x = self.norm(x)
        x = self.lin(x)
        x = nn.ReLU()(x)
        return x + res


def compute_distances(x1: Tensor, x2: Tensor) -> Tensor:
    norm1 = ei.reduce(x1**2, "n1 d -> n1 1", "sum")
    norm2 = ei.reduce(x2**2, "n2 d -> 1 n2", "sum")
    dot_prod = torch.matmul(x1, x2.t())
    return norm1 + norm2 - 2 * dot_prod


class VectorQuantizer1D(nn.Module):
    """Quantizes input sequences of vectors using a codebook"""

    def __init__(
        self,
        codebook_size: int,
        dim: int,
    ):
        super().__init__()
        self.codebook = nn.Embedding(num_embeddings=codebook_size, embedding_dim=dim)
        self.codebook_size = codebook_size
        self.dim = dim

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        bsize = x.size(0)
        seq_len = x.size(1)
        # flatten
        flat_x = ei.rearrange(x, "b l d -> (b l) d")

        # compute distances
        distances = compute_distances(flat_x, self.codebook.weight)

        # quantize
        indices = torch.argmin(distances, dim=1)
        z_quantized = self.codebook(indices)

        # unflatten
        z_quantized = ei.rearrange(
            z_quantized, "(b l) d -> b l d", b=bsize, l=seq_len, d=self.dim
        )
        return z_quantized, ei.rearrange(indices, "(b l) -> b l", b=bsize, l=seq_len)

    def sample_uniform(self, n_samples: int, seq_len: int) -> Tensor:
        rd_idxs = torch.randint(
            low=0, high=self.codebook_size, size=(n_samples * seq_len,)
        )
        quantized = self.codebook(rd_idxs)
        return ei.rearrange(
            quantized, "(n l) d -> n l d", n=n_samples, l=seq_len, d=self.dim
        )
