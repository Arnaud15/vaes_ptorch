"""Very Deep VAE implementation with FeedForward blocks.

- paper: https://arxiv.org/abs/2011.10650
- official code (CNN blocks): https://github.com/openai/vdvae
"""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor


def gaussian_analytical_kl(
    mu1: Tensor, mu2: Tensor, logsigma1: Tensor, logsigma2: Tensor
) -> Tensor:
    """KL divergence between two multivariate gaussians with independent components."""
    return (
        -0.5
        + logsigma2
        - logsigma1
        + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2)
    )


def draw_gaussian_diag_samples(mu: Tensor, logsigma: Tensor) -> Tensor:
    """Sampling from a gaussian with the reparameterization trick."""
    eps = torch.empty_like(mu).normal_(0.0, 1.0)
    return torch.exp(logsigma) * eps + mu


class Block(nn.Module):
    """1-hidden-layer MLP with optional residual connection."""

    def __init__(
        self, in_dim: int, h_dim: int, out_dim: int, residual=False,
    ):
        super().__init__()
        self.residual = residual
        self.layers = nn.Sequential(
            nn.Linear(in_dim, h_dim,),
            nn.SiLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.SiLU(inplace=True),
            nn.Linear(h_dim, out_dim),
        )
        if self.residual:
            self.proj_out = (
                nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
            )
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == self.in_dim, x.shape
        x_hat = self.layers(x)
        out = self.proj_out(x) + x_hat if self.residual else x_hat
        assert out.shape[-1] == self.out_dim, x.shape
        return out


class DecoderBlock(nn.Module):
    """Decoder block in the VD-VAE architecture."""

    def __init__(
        self, in_dim: int, out_dim: int,
    ):
        super().__init__()
        ### PRIOR NETWORK
        # Takes a transformed z_prev as input
        # Outputs 3 vectors
        # - "autoregressive features" (z | z_prev)
        # - mu_z | z_prev prior
        # - logsigma_z | z_prev prior
        self.prior = Block(
            in_dim=in_dim, h_dim=out_dim, out_dim=out_dim * 3, residual=False
        )
        ### POSTERIOR NETWORK
        # Takes (transformed z_prev, x_context) as inputs
        # Outputs 2 vectors
        # - mu_z | (x, z_prev) posterior
        # - logsigma_z | (x, z_prev) posterior
        self.posterior = Block(
            in_dim=in_dim * 2, h_dim=out_dim, out_dim=out_dim * 2, residual=False
        )
        ### PROJ PREV Z CONTEXT MLP
        # Takes the prev context as input
        # Outputs a single vector
        self.proj_prev_z = Block(
            in_dim=in_dim, h_dim=out_dim, out_dim=out_dim, residual=False
        )
        ### PROJ Z SAMPLES MLP
        # Takes a z_sample as input
        # Outputs a single vector
        self.proj_z_sample = Block(
            in_dim=out_dim, h_dim=out_dim, out_dim=out_dim, residual=False
        )
        ### TRANSFORM MLP
        # Takes (z | z_prev) features += transformed(z_sample) as single input
        # Outputs a single vector
        self.transform = Block(
            in_dim=out_dim, h_dim=out_dim, out_dim=out_dim, residual=True
        )
        self.out_dim = out_dim
        self.in_dim = in_dim

    def forward(
        self, prev_zs_context: Tensor, activations: Dict[int, Tensor]
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        x_context = activations[self.in_dim]
        assert x_context.shape[-1] == self.in_dim, x_context.shape
        # prior pass
        z_features, mu_z, logsigma_z = self.prior(prev_zs_context).chunk(
            chunks=3, dim=-1
        )
        # posterior pass
        mu_zx, logsigma_zx = self.posterior(
            torch.cat([prev_zs_context, x_context], dim=-1)
        ).chunk(chunks=2, dim=-1)
        # kl divergence computation
        kl = gaussian_analytical_kl(
            mu1=mu_zx, mu2=mu_z, logsigma1=logsigma_zx, logsigma2=logsigma_z
        ).sum()
        # z_features
        z_features = z_features + self.proj_prev_z(prev_zs_context)
        z_sample = draw_gaussian_diag_samples(mu=mu_zx, logsigma=logsigma_zx)
        # The actual z sampled matters
        z_features += self.proj_z_sample(z_sample)
        z_features = self.transform(z_features)
        assert z_features.shape[-1] == self.out_dim, z_features.shape
        return z_features, dict(kl=kl)

    def forward_uncond(self, prev_zs_context: Tensor) -> Tensor:
        z_features, mu_z, logsigma_z = self.prior(prev_zs_context).chunk(
            chunks=3, dim=-1
        )
        # z_features
        z_features = z_features + self.proj_prev_z(prev_zs_context)
        z_sample = draw_gaussian_diag_samples(mu=mu_z, logsigma=logsigma_z)
        # The actual z sampled matters
        z_features += self.proj_z_sample(z_sample)
        z_features = self.transform(z_features)
        assert z_features.shape[-1] == self.out_dim, z_features.shape
        return z_features


class Encoder(nn.Module):
    """Bottom-up Encoder network for the VD-VAE."""

    def __init__(self, dims: List[int]):
        super().__init__()
        assert len(dims) >= 2
        self.out_dims = dims[1:]
        self.blocks = nn.ModuleList()
        for dim_ix in range(len(dims) - 1):
            in_dim, out_dim = dims[dim_ix], dims[dim_ix + 1]
            assert out_dim < in_dim, (in_dim, out_dim)
            self.blocks.append(
                Block(in_dim=in_dim, h_dim=out_dim, out_dim=out_dim, residual=True)
            )
        assert len(self.out_dims) == len(self.blocks)

    def forward(self, x: Tensor) -> Dict[int, Tensor]:
        activations = {}
        for block, out_dim in zip(self.blocks, self.out_dims):
            x = block(x)
            activations[out_dim] = x
        return activations


class Decoder(nn.Module):
    """Top-down Decoder network for the VD-VAE."""

    def __init__(self, dims: List[int]):
        super().__init__()
        assert len(dims) >= 2
        self.in_dim = dims[0]
        self.blocks = nn.ModuleList()
        for dim_ix in range(len(dims) - 1):
            in_dim, out_dim = dims[dim_ix], dims[dim_ix + 1]
            assert out_dim > in_dim, (in_dim, out_dim)
            self.blocks.append(DecoderBlock(in_dim=in_dim, out_dim=out_dim))

    def forward(self, activations):
        stats = []
        any_input = next(iter(activations.values()))
        device, bsize = any_input.device, any_input.shape[0]
        prev_zs_context = torch.zeros(
            (bsize, self.in_dim), device=device, dtype=torch.float32
        )
        for block in self.blocks:
            prev_zs_context, block_stats = block(prev_zs_context, activations)
            stats.append(block_stats)
        return prev_zs_context, stats

    def sample(self, placeholder_x: Tensor) -> Tensor:
        device, bsize = placeholder_x.device, placeholder_x.shape[0]
        x = torch.zeros((bsize, self.in_dim), device=device, dtype=torch.float32)
        for block in self.blocks:
            x = block.forward_uncond(x)
        return x


class VDVAE(nn.Module):
    """Simple VD-VAE network with MLP blocks."""

    def __init__(self, in_dim: int, bottom_up_sizes: List[int]):
        super().__init__()
        assert len(bottom_up_sizes) > 0
        full_dims = [in_dim] + bottom_up_sizes
        self.encoder = Encoder(full_dims)
        self.decoder = Decoder(full_dims[::-1])

    def forward(
        self, x: Tensor, kl_scale: float = 1.0
    ) -> Tuple[Tensor, Dict[str, float]]:
        bsize = x.shape[0]
        activations = self.encoder.forward(x)
        x_hat_z, stats = self.decoder.forward(activations)
        rec_loss = torch.nn.functional.mse_loss(
            input=x_hat_z, target=x, reduction="mean"
        )
        kl_loss = 0.0
        for statdict in stats:
            assert statdict["kl"].ndim == 0
            kl_loss += statdict["kl"]
        kl_loss /= bsize
        elbo = rec_loss + kl_loss * kl_scale
        return elbo, dict(rec_loss=rec_loss.item(), kl_loss=kl_loss.item())

    def sample(self, placeholder_x: Tensor) -> Tensor:
        return self.decoder.sample(placeholder_x)

    def reconstruct(self, x: Tensor) -> Tensor:
        activations = self.encoder.forward(x)
        x_hat, _ = self.decoder.forward(activations)
        return x_hat


if __name__ == "__main__":
    import numpy as np

    import vaes_ptorch.plot as plot
    import vaes_ptorch.utils as ut
    import vaes_ptorch.annealing as annealing

    # params
    DSET_SIZE = 16384
    DIM = 16

    BATCH_SIZE = 128
    LR = 1e-3
    N_EPOCHS = 20

    DIMS_HIERARCHY = list(range(DIM - 1, 0, -1))

    TARGET_LAMBDA = 2.5  # 2.5 for [12, 6, 3]

    # VD-VAE init
    net = VDVAE(in_dim=DIM, bottom_up_sizes=DIMS_HIERARCHY)

    # data generation
    P = torch.zeros((2, DIM))
    P[0, ::2] = 1
    P[1, 1::2] = 1
    data_x = torch.linspace(0, 2 * np.pi, 7)[:-1]
    data_x = torch.stack((torch.cos(data_x), torch.sin(data_x)), dim=1)
    data_x = data_x[None] + torch.randn((DSET_SIZE, data_x.shape[0], 2)) * 0.1
    data_x = data_x.view(-1, 2)
    plot.plot_points_series([data_x[i::6].numpy() for i in range(6)])

    opt = torch.optim.Adam(net.parameters(), lr=LR)
    kl_scale_manager = annealing.SoftFreeBits(target_lambda=TARGET_LAMBDA)
    for epoch in range(N_EPOCHS):

        info = {}

        for _ in range(0, DSET_SIZE, BATCH_SIZE):
            idx = torch.randint(data_x.shape[0], size=(BATCH_SIZE,))
            batch = data_x[idx] @ P
            loss, loss_dict = net(batch, kl_scale=kl_scale_manager.get_scale())

            opt.zero_grad()
            loss.backward()
            opt.step()
            kl_scale_manager.step(loss_dict["kl_loss"])

            ut.update_info_dict(info, obs={"vdvae_loss": loss.item()} | loss_dict)
        print(ut.print_info_dict(info))

    # VD-VAE reconstruction
    net.eval()
    with torch.no_grad():
        x_sample = net.reconstruct(data_x @ P)
        plot.plot_points_series([x_sample.numpy()])

    # VD-VAE sampling
    net.eval()
    with torch.no_grad():
        placeHolder = torch.zeros_like(data_x @ P)
        x_sample = net.sample(placeHolder)
        plot.plot_points_series([x_sample.numpy()])
