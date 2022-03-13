"""The Gaussian VAE class."""
import enum
import math
from typing import Dict, Tuple

import torch
import torch.distributions as tdist
import torch.nn as nn
from torch import Tensor
from torch.distributions.distribution import Distribution

import vaes_ptorch.proba as proba
import vaes_ptorch.utils as ut

# TODO -> move to einops


class Divergence(enum.Enum):
    """Available divergence measures for use in the VAE loss."""

    KL = enum.auto()
    MMD = enum.auto()


class GaussianVAE(nn.Module):
    """Gaussian VAE: the classic VAE with a unit multivariate gaussian prior."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int,
        obs_model: proba.ObsModel = proba.ObsModel.UnitGaussian,
    ):
        super(GaussianVAE, self).__init__()
        assert latent_dim > 0, f"found nonpositive latent dim: {latent_dim}"
        self.latent_dim = latent_dim
        self.obs_model = obs_model
        self.encoder = encoder
        self.decoder = decoder
        self.prior = tdist.normal.Normal(loc=0.0, scale=1.0)

    def sample_prior(self, n_samples: int) -> Tensor:
        """Sample from the prior using the reparameterization trick"""
        return self.prior.rsample((n_samples, self.latent_dim))

    def encode(self, x: Tensor) -> tdist.normal.Normal:
        """Encode `x` to obtain the `q(z | x)` posterior distribution."""
        q_z_given_x_params = self.encoder(x)
        return proba.to_gaussian_dist(
            q_z_given_x_params,
            dim=self.latent_dim,
        )

    def decode(self, z: Tensor) -> Distribution:
        """Decode `z` to obtain the `p(x | z)` reconstruction distribution."""
        p_x_given_z_params = self.decoder(z)
        return proba.params_to_dist(p_x_given_z_params, obs_model=self.obs_model)

    def forward(self, x: Tensor) -> Tuple[tdist.normal.Normal, Distribution]:
        """Forward pass through a Gaussian VAE.

        1. compute the posterior distribution given x
        2. sample z from the posterior using the reparameterization trick
        3. compute the reconstruction distribution given z"""
        q_z_given_x = self.encode(x)
        z_sample = q_z_given_x.rsample()
        p_x_given_z = self.decode(z_sample)
        return (q_z_given_x, p_x_given_z)

    def divergence_loss(
        self, q_z_given_x: tdist.normal.Normal, div_type: Divergence
    ) -> Tensor:
        """Compute the divergence term of the ELBO loss.

        Supports both vanilla VAEs and MMD-VAEs."""
        if div_type == Divergence.KL:
            # vanilla VAE: KL divergence with the unit prior
            bsize = q_z_given_x.mean.shape[0]
            return tdist.kl.kl_divergence(q_z_given_x, self.prior).sum() / bsize
        elif div_type == Divergence.MMD:
            # MMD VAE, compute MMD(q(z) | p(z))
            # z samples from the variational posterior distribution
            z_posterior_samples = q_z_given_x.rsample()
            # z samples from the prior distribution
            device = z_posterior_samples.device
            z_prior_samples = self.sample_prior(
                n_samples=z_posterior_samples.size(0)
            ).to(device)
            return proba.mmd_rbf(
                samples_p=z_posterior_samples,
                samples_q=z_prior_samples,
            )
        else:
            raise NotImplementedError(
                f"unrecognized divergence type: {div_type}, expected [KL | MMD]"
            )

    def loss(
        self, x: Tensor, div_type: Divergence, div_scale: float = 1.0
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Computes the ELBO (Evidence Lower Bound) loss for a Gaussian VAE.

        Allows
        - the re-scaling of the divergence term via the `div_scale` parameter
          (can be used to implement a beta-VAE, and to stabilize training more
          generally).
        - the specification of the divergence term (the KL divergence
          corresponds to a vanilla or beta VAE, the MMD over `z` samples
          corresponds to a MMD-VAE / InfoVAE).

        Returns:
        - the average loss over the input batch
        - the two components of the loss as floats: negative log likelihood and
          divergence terms.

        References:
        - original VAE paper: https://arxiv.org/abs/1312.6114.
        - beta-VAE paper: https://openreview.net/forum?id=Sy2fzU9gl
        - InfoVAE paper: https://arxiv.org/abs/1706.02262
        """

        assert div_scale >= 0.0
        # assume that the first dimension is the batch0.0
        batch_size = x.size(0)
        assert x.dim() > 1
        q_z_given_x, p_x_given_z = self.forward(x)
        reconstruction_term = -p_x_given_z.log_prob(x).sum() / batch_size
        div_term = self.divergence_loss(q_z_given_x, div_type=div_type)
        loss = reconstruction_term + div_scale * div_term
        return (
            loss,
            {
                "loss": loss.item(),
                "nll": reconstruction_term.item(),
                "div": div_term.item(),
            },
        )

    def nll_is(self, x: Tensor, n_samples: int = 100) -> float:
        """Estimate the negative log likelihood of a VAE on a batch of
        observations `x` using importance sampling.
        """
        assert x.dim() > 1  # assume that the first dimension is the batch
        bsize = x.size(0)
        x_dims = x.shape[1:]

        q_z_given_x = self.encode(x)
        z_samples = q_z_given_x.rsample((n_samples,))
        assert z_samples.shape == (n_samples, bsize, self.latent_dim)

        z_nll_q = -q_z_given_x.log_prob(z_samples)
        assert z_nll_q.shape == (n_samples, bsize)
        if torch.any(torch.isinf(z_nll_q)):
            print("warning: infinite value in z samples nll | q")

        z_nll_prior = -self.prior.log_prob(z_samples)
        assert z_nll_prior.shape == (n_samples, bsize)
        if torch.any(torch.isinf(z_nll_prior)):
            print("warning: infinite value in z samples nll | prior")

        p_x_given_z = self.decode(z_samples)
        reconstruction_nll = -p_x_given_z.log_prob(x)
        assert reconstruction_nll.shape == (n_samples * bsize, *x_dims)
        reconstruction_nll = reconstruction_nll.sum(dim=list(range(1, 1 + len(x_dims))))
        assert reconstruction_nll.shape == (n_samples * bsize,)
        reconstruction_nll = reconstruction_nll.reshape(n_samples, bsize)

        if torch.any(torch.isinf(reconstruction_nll)):
            print("warning: infinite value in reconstruction nll")

        log_likelihood_estimates = torch.logsumexp(
            z_nll_q - reconstruction_nll - z_nll_prior,
            dim=0,
            keepdim=False,
        ) - math.log(n_samples)
        assert log_likelihood_estimates.shape == (bsize,)

        if torch.any(torch.isinf(log_likelihood_estimates)):
            print("warning: infinite value in log likelihood estimates")

        return -log_likelihood_estimates.mean().item() / ut.bits_per_dim_multiplier(
            list(x_dims)
        )


if __name__ == "__main__":
    import numpy as np

    import vaes_ptorch.models as models
    import vaes_ptorch.plot as plot
    import vaes_ptorch.utils as ut

    # params
    DSET_SIZE = 16384
    DIM = 16

    BATCH_SIZE = 64
    LR = 1e-3
    N_EPOCHS = 30

    LATENT_DIM = 1
    N_LAYERS = 5
    H_DIM = 64

    DIV_SCALE = 0.1
    DIV_TYPE = Divergence.KL

    # model and optimizer init
    encoder = models.get_mlp(
        in_dim=DIM, out_dim=2 * LATENT_DIM, h_dim=H_DIM, n_hidden=N_LAYERS
    )
    decoder = models.get_mlp(
        in_dim=LATENT_DIM, out_dim=DIM, h_dim=H_DIM, n_hidden=N_LAYERS
    )
    net = GaussianVAE(
        encoder=encoder,
        decoder=decoder,
        latent_dim=LATENT_DIM,
    )
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    # data generation
    P = torch.zeros((2, DIM))
    P[0, ::2] = 1
    P[1, 1::2] = 1
    data_x = torch.linspace(0, 2 * np.pi, 7)[:-1]
    data_x = torch.stack((torch.cos(data_x), torch.sin(data_x)), dim=1)
    data_x = data_x[None] + torch.randn((DSET_SIZE, data_x.shape[0], 2)) * 0.1
    data_x = data_x.view(-1, 2)
    plot.plot_points_series([data_x[i::6].numpy() for i in range(6)])

    # VAE reconstruction training
    net.train()
    for epoch in range(N_EPOCHS):

        info = {}

        for _ in range(0, DSET_SIZE, BATCH_SIZE):
            idx = torch.randint(data_x.shape[0], size=(BATCH_SIZE,))
            batch = data_x[idx] @ P
            loss, loss_info = net.loss(batch, div_type=DIV_TYPE, div_scale=DIV_SCALE)

            opt.zero_grad()
            loss.backward()
            opt.step()

            ut.update_info_dict(info, obs=loss_info)

        print(f"Epoch {epoch + 1} | Info: {ut.print_info_dict(info)}.")

    # Testing the reconstruction
    net.eval()
    with torch.no_grad():
        series = []
        for i in range(6):
            full_batch = data_x[i::6] @ P
            _, p_x_given_z = net(full_batch)
            series.append(p_x_given_z.mean.numpy())
        plot.plot_points_series(series)

    # uniform sampling
    with torch.no_grad():
        prior_samples = net.sample_prior(DSET_SIZE)
        x_samples = net.decode(prior_samples).mean.numpy()
        plot.plot_points_series([x_samples])
