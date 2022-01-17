"""The main VAE class"""
import enum
import math
from collections import namedtuple
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

import vaes_ptorch.args as args
import vaes_ptorch.models as models
import vaes_ptorch.proba as proba
import vaes_ptorch.utils as ut

VaeOutput = namedtuple("VaeOutput", ["z_params", "z_sample", "x_params",])

ElboLoss = namedtuple("ElboLoss", ["loss", "nll", "div"])


class Divergence(enum.Enum):
    """Divergence measures for use in the VAE loss"""

    KL = enum.auto()
    MMD = enum.auto()


class GaussianVAE(nn.Module):
    """
    Class for VAE models with the following constraints:
    - the prior distribution over the latent space is a unit multivariate
      normal
    - the decoder maps latent variables to parameters of a probability
      distribution over the observation space as specified by a
      `proba.StatsModel`. Two stats model are currently supported:
        (1) Gaussian: a multivariate gaussian distribution with independent
        components
        (2) Bernoulli: probabilities for independent bernoulli random variables
    """

    def __init__(
        self,
        encoder: models.GaussianNN,
        decoder: nn.Module,
        latent_dim: int,
        stats_model: proba.StatsModel,
    ):
        super(GaussianVAE, self).__init__()
        assert latent_dim > 0, f"found nonpositive latent dim: {latent_dim}"
        self.latent_dim = latent_dim
        self.stats_model = stats_model
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x: Tensor) -> proba.NormalParams:
        return self.encoder(x)

    def decode(self, x: Tensor) -> Any:
        return self.decoder(x)

    def sample_posterior(
        self, posterior_params: proba.NormalParams, n_samples: int = 1
    ) -> Tensor:
        return proba.sample_gaussian(posterior_params, n_samples=n_samples)

    def sample_prior(self, n_samples: int, device: torch.device) -> Tensor:
        assert (
            n_samples > 0
        ), f"found nonpositive number of samples to generate: {n_samples}"
        return proba.sample_gaussian(
            proba.NormalParams(
                mu=torch.zeros(self.latent_dim, device=device),
                var=torch.ones(self.latent_dim, device=device),
            ),
            n_samples=n_samples,
        )

    def sample_obs(self, params: Any, n_samples: int) -> Tensor:
        return self.stats_model.sample_obs(params=params, n_samples=n_samples)

    def generate_x(self, n_samples: int, device: torch.device) -> Tensor:
        assert (
            n_samples > 0
        ), f"found nonpositive number of samples to generate: {n_samples}"
        prior_samples = self.sample_prior(n_samples, device)
        x_params = self.decode(prior_samples)
        return self.sample_obs(x_params, n_samples=1)

    def forward(self, x: Tensor) -> VaeOutput:
        z_params = self.encode(x)
        z_sample = self.sample_posterior(z_params)
        x_params = self.decode(z_sample)
        return VaeOutput(z_params=z_params, z_sample=z_sample, x_params=x_params)

    def reconstruction_loss(self, x: Tensor, params: Any) -> Tensor:
        batch_size = x.size(0)
        return self.stats_model.compute_nll(obs=x, params=params).sum() / batch_size

    def divergence_loss(
        self, z_params: proba.NormalParams, div_type: Divergence
    ) -> Tensor:
        if div_type == Divergence.KL:
            return proba.gaussian_kl(
                # variational posterior parameters
                left_mu=z_params.mu,
                left_var=z_params.var,
                # prior distribution parameters
                right_mu=torch.zeros_like(z_params.mu),
                right_var=torch.ones_like(z_params.var),
            )
        elif div_type == Divergence.MMD:
            # z samples from the variational posterior distribution
            # z ~ p_data(x) * q_{phi}(z | x)
            z_posterior_samples = proba.sample_gaussian(z_params, n_samples=1)
            # z samples from the prior distribution
            z_prior_samples = self.sample_prior(
                n_samples=z_posterior_samples.size(0), device=z_posterior_samples.device
            )
            return proba.mmd_rbf(
                samples_p=z_posterior_samples, samples_q=z_prior_samples,
            )
        else:
            raise ValueError(
                f"unrecognized divergence type: {div_type}, expected [KL | MMD]"
            )

    def compute_elbo(
        self, x: Tensor, div_type: Divergence, div_scale: float = 1.0
    ) -> ElboLoss:
        """Computes ELBO (Evidence Lower Bound) loss variatns for the Gaussian
        VAE.

        Allows
        - the re-scaling of the divergence term via the `div_scale` parameter
          (can be used to implement a beta-VAE, and to stabilize training more
          generally).
        - the specification of the divergence term (the KL divergence
          corresponds to a vanilla or beta VAE, the MMD over `z` samples
          corresponds to an InfoVAE).

        Returns:
        - the mean loss over the input batch
        - the two components of the loss as floats: negative log likelihood and
          divergence terms.

        References:
        - original VAE paper: https://arxiv.org/abs/1312.6114.
        - beta-VAE paper: https://openreview.net/forum?id=Sy2fzU9gl
        - InfoVAE paper: https://arxiv.org/abs/1706.02262 
        """
        assert div_scale >= 0.0
        assert x.dim() > 1  # assume that the first dimension is the batch
        output = self.forward(x)
        reconstruction_loss = self.reconstruction_loss(x=x, params=output.x_params)
        div_term = self.divergence_loss(output.z_params, div_type=div_type)
        return ElboLoss(
            loss=reconstruction_loss + div_scale * div_term,
            nll=reconstruction_loss.item(),
            div=div_term.item(),
        )

    def nll_is(self, x: Tensor, n_samples: int = 100) -> float:
        """Estimate the negative log likelihood of a VAE on a batch of
        observations `x` using importance sampling.
        """
        assert x.dim() > 1  # assume that the first dimension is the batch
        bsize = x.size(0)
        x_dims = x.shape[1:]

        out = self.forward(x)
        z_samples = self.sample_posterior(out.z_params, n_samples=n_samples)
        assert z_samples.shape == (n_samples, bsize, self.latent_dim)

        z_params_dup = proba.NormalParams(
            mu=torch.tile(out.z_params.mu.unsqueeze(0), (n_samples, 1, 1)),
            var=torch.tile(out.z_params.var.unsqueeze(0), (n_samples, 1, 1)),
        )
        assert z_params_dup.mu.shape == (n_samples, bsize, self.latent_dim)
        assert z_params_dup.var.shape == (n_samples, bsize, self.latent_dim)
        posterior_nll = (
            proba.GaussianModel()
            .compute_nll(obs=z_samples, params=z_params_dup)
            .sum(-1)
        )
        assert posterior_nll.shape == (n_samples, bsize)
        if torch.any(torch.isinf(posterior_nll)):
            print("warning: infinite value in posterior")

        prior_nll = (
            proba.GaussianModel()
            .compute_nll(
                obs=z_samples,
                params=proba.NormalParams(
                    mu=torch.zeros_like(z_samples), var=torch.ones_like(z_samples)
                ),
            )
            .sum(-1)
        )
        assert prior_nll.shape == (n_samples, bsize)
        if torch.any(torch.isinf(prior_nll)):
            print("warning: infinite value in prior")

        x_params = self.decode(torch.flatten(z_samples, start_dim=0, end_dim=1))
        x_target = torch.tile(
            x.unsqueeze(0), tuple([n_samples] + [1] * x.dim())
        ).reshape(n_samples * bsize, *x_dims)
        reconstruction_nll = self.stats_model.compute_nll(obs=x_target, params=x_params)
        assert reconstruction_nll.shape == (n_samples * bsize, *x_dims)
        reconstruction_nll = reconstruction_nll.sum(dim=list(range(1, 1 + len(x_dims))))
        assert reconstruction_nll.shape == (n_samples * bsize,)
        reconstruction_nll = reconstruction_nll.reshape(n_samples, bsize)

        if torch.any(torch.isinf(reconstruction_nll)):
            print("warning: infinite value in reconstruction_nll")

        log_likelihood_estimates = torch.logsumexp(
            posterior_nll - reconstruction_nll - prior_nll, dim=0, keepdim=False,
        ) - math.log(n_samples)
        assert log_likelihood_estimates.shape == (bsize,)

        if torch.any(torch.isinf(log_likelihood_estimates)):
            print("warning: infinite value in log likelihood estimates")

        return -log_likelihood_estimates.mean().item() / ut.bits_per_dim_multiplier(
            list(x_dims)
        )
