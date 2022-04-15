"""Neural network modules used in experiments."""
from typing import Tuple

import einops as ei
import torch
import torch.nn as nn
from torch import Tensor


def get_mlp(in_dim: int, out_dim: int, h_dim: int, n_hidden: int) -> nn.Module:
    """Build a Multi Layer Perceptron (MLP) with residual connections, switch
    activations, Layer Normalization and a fixed hidden size."""
    assert n_hidden >= 0, n_hidden
    if not n_hidden:
        return nn.Linear(in_dim, out_dim)
    else:
        return nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.SiLU(),
            *[ResBlock(dim=h_dim) for _ in range(n_hidden)],
            nn.LayerNorm(h_dim),
            nn.Linear(h_dim, out_dim),
        )


class ResBlock(nn.Module):
    """Fully Connected residual block with Layer Norm and switch activation."""

    def __init__(self, dim: int):
        super(ResBlock, self).__init__()
        assert dim > 0, dim
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.lin = nn.Linear(dim, dim)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        assert x.size(-1) == self.dim, (x.size(), self.dim)
        res = x
        x = self.norm(x)
        x = self.lin(x)
        x = self.act(x)
        return x + res


def compute_distances(x1: Tensor, x2: Tensor) -> Tensor:
    norm1 = ei.reduce(x1 ** 2, "n1 d -> n1 1", "sum")
    norm2 = ei.reduce(x2 ** 2, "n2 d -> 1 n2", "sum")
    dot_prod = torch.matmul(x1, x2.t())
    return norm1 + norm2 - 2 * dot_prod


class VectorQuantizer1D(nn.Module):
    """Quantizes input sequences of vectors using a codebook"""

    def __init__(
        self, codebook_size: int, dim: int,
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


class SeqModel(nn.Module):
    """Simple 1D autoregressive model made up of a GRU and a linear projection
    head"""

    def __init__(self, input_size: int, hidden_size: int, codebook_size: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bias=True,
            batch_first=True,
        )
        self.proj_head = nn.Linear(hidden_size, codebook_size)
        self.dim = hidden_size

    def forward(self, x: Tensor) -> Tensor:
        h_states, _ = self.gru(x)
        h_0 = torch.zeros(x.size(0), 1, self.dim)
        prev_h_states = torch.cat([h_0, h_states[:, :-1, :]], dim=1)
        assert h_states.shape == prev_h_states.shape
        return self.proj_head(
            prev_h_states
        )  # concatenate a 0 at the beginning and ignore the last hidden state


class EmbeddingFourier(nn.Module):
    MAX_LEN = 10_000

    def __init__(self, n_pos: int, dim: int):
        super().__init__()
        pos_embed = self.make_position_embedding(n_pos=n_pos, dim=dim)
        self.register_buffer("pos_embed", pos_embed)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim), torch.nn.SiLU(), torch.nn.Linear(dim, dim)
        )
        self.dim = dim

    @classmethod
    def make_position_embedding(cls, n_pos: int, dim: int):
        assert dim % 2 == 0
        exponents = (torch.arange(0, dim, 2) / dim).float()
        dim_shares = 1.0 / (cls.MAX_LEN ** exponents)
        pos = torch.arange(n_pos)
        cos_emb = torch.cos(pos.unsqueeze(-1) * dim_shares.unsqueeze(0))
        sin_emb = torch.sin(pos.unsqueeze(-1) * dim_shares.unsqueeze(0))
        return torch.cat([sin_emb, cos_emb], dim=1)

    def forward(self, t: Tensor) -> Tensor:
        # Assume n_pos >= t >= 1, long tensor of shape (B, 1)
        bsize = t.shape[0]
        assert t.shape == (bsize, 1)
        embeds = torch.index_select(self.pos_embed, dim=0, index=(t - 1).squeeze(-1))
        out = self.mlp(embeds)
        assert out.shape == (bsize, self.dim)
        return out


def make_fourier_features(in_dim, output_dim, scale):
    assert output_dim % 2 == 0
    return scale * torch.randn(in_dim, output_dim // 2)


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, scale: float = 1.0):
        super().__init__()
        self.register_buffer("b", make_fourier_features(in_dim, out_dim, scale=scale))
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        assert x.shape[-1] == self.in_dim
        proj = torch.matmul(x, self.b)
        out = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        assert out.shape[-1] == self.out_dim
        return out
