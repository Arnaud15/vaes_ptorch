import enum
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .utils import gaussian_kl, gaussian_nll, mmd_rbf, sample_gaussian
from .vae import GaussianVAE, VaeOutput


class Likelihood(enum.Enum):
    Gaussian = enum.auto()
    Bernoulli = enum.auto()


class Divergence(enum.Enum):
    KL = enum.auto()
    MMD = enum.auto()


def elbo_loss(
    x: Tensor,
    vae_output: VaeOutput,
    nll_type: Likelihood,
    div_type: Divergence,
    div_scale: float = 1.0,
) -> Tuple[Tensor, Tuple[float, float]]:
    """Computes the ELBO (Evidence Lower Bound) loss for a Gaussian VAE.

    Assumes that
    - The prior distribution over latents is a vector of independent
    unit gaussian distributions.
    - The vae decoder parameterizes either:
        (1) a multivariate gaussian distribution with independent components
        (pick a `Gaussian` likelihood)
      or
        (2) a multivariate distribution where with independent bernoulli
        components (pick a `Bernoulli` likelihood).

    Allows
    - the re-scaling of the KL divergence via the `div_scale` parameter (can be
      used directly to implement a beta-VAE, and to stabilize training more
      generally).
    - the specification of two types of log likelihood functions: gaussian or
      bernoulli.
    - the specification of the divergence term (the KL divergence corresponds
      to a vanilla VAE, the MMD over `z` samples corresponds to MMD-VAE).

    Returns:
    - the total ELBO loss
    - the two components of the ELBO: negative log likelihood and divergence
      terms.

    References:
    - original VAE paper: https://arxiv.org/abs/1312.6114.
    - beta-VAE paper: https://openreview.net/forum?id=Sy2fzU9gl
    - MMD-VAE paper (InfoVAE): https://arxiv.org/abs/1706.02262 
    """
    assert div_scale >= 0.0
    assert x.dim() > 1  # assume the first dimension is the batch
    batch_size = x.size(0)
    assert batch_size > 0

    if nll_type == Likelihood.Gaussian:
        nll = nn.GaussianNLLLoss(reduction="sum", eps=1.0)(
            vae_output.mu_x, target=x, var=vae_output.sig_x
        )
    elif nll_type == Likelihood.Bernoulli:
        nll = nn.BCEWithLogitsLoss(reduction="sum")(vae_output.mu_x, target=x)
    else:
        raise ValueError(
            f"incorrect negative log likelihood type, found {nll_type} but expected Gaussian or Bernoulli"
        )
    nll /= batch_size
    assert not nll.dim(), nll.dim()

    if div_type == Divergence.KL:
        div = gaussian_kl(
            left_mu=vae_output.mu_z,
            left_sig=vae_output.sig_z,
            right_mu=torch.zeros_like(vae_output.mu_z),
            right_sig=torch.ones_like(vae_output.sig_z),
        )
    elif div_type == Divergence.MMD:
        # z samples from the variational posterior distribution
        # z ~ p_data(x) * q_{phi}(z | x)
        z_samples = (
            torch.randn_like(vae_output.mu_z) * torch.sqrt(vae_output.sig_z)
            + vae_output.mu_z
        )
        # z samples from the prior distribution
        z_prior = torch.randn_like(vae_output.mu_z)
        div = mmd_rbf(samples_p=z_samples, samples_q=z_prior,)
    else:
        raise ValueError(
            f"incorrect divergence type, found {div_type} but expected PosteriorPriorKL (Vanilla VAE and beta-VAE) or PriorMMD (MMD-VAE)"
        )
    assert not div.dim(), div.dim()

    return nll + div_scale * div, (nll.item(), div.item())


def nll_is(
    x: torch.Tensor, vae_nn: GaussianVAE, nll_type: Likelihood, n_samples: int = 100,
) -> float:
    """Estimate the negative log likelihood of a VAE on a batch of datapoints `x` using importance sampling."""
    bsize = x.size(0)
    x_dims = x.shape[1:]
    dims_prod = torch.prod(torch.tensor(list(x_dims))).item()
    bit_per_dim_normalizer = math.log(2.0) * dims_prod
    mu_z, sig_z = vae_nn.encoder(x)
    latent_dim = mu_z.size(-1)
    assert mu_z.shape == (bsize, latent_dim)
    assert sig_z.shape == (bsize, latent_dim)

    z_samples = sample_gaussian(mu_z, sig_z, n_samples=n_samples)
    assert z_samples.shape == (n_samples, bsize, latent_dim)

    approx_posterior_nll = gaussian_nll(obs=z_samples, mean=mu_z, var=sig_z,).sum(-1)
    if torch.any(torch.isinf(approx_posterior_nll)):
        print("warning: infinite value in posterior")

    prior_nll = gaussian_nll(
        mean=torch.zeros_like(z_samples), obs=z_samples, var=torch.ones_like(z_samples),
    ).sum(-1)
    if torch.any(torch.isinf(prior_nll)):
        print("warning: infinite value in prior")

    mu_x, sig_x = vae_nn.decoder(torch.flatten(z_samples, start_dim=0, end_dim=1))
    assert mu_x.size(0) == bsize * n_samples
    assert sig_x.size(0) == bsize * n_samples

    mu_x = mu_x.reshape(n_samples, bsize, *x_dims)
    sig_x = sig_x.reshape(n_samples, bsize, *x_dims)
    if nll_type == Likelihood.Gaussian:
        reconstruction_nll = gaussian_nll(mean=mu_x, obs=x, var=sig_x)
    elif nll_type == Likelihood.Bernoulli:
        x_target = torch.tile(x.unsqueeze(0), tuple([n_samples] + [1] * x.dim()))
        assert x_target.shape[2:] == x_dims
        assert x_target.size(0) == n_samples
        assert x_target.size(1) == bsize
        reconstruction_nll = nn.BCEWithLogitsLoss(reduction="none")(
            mu_x, target=x_target
        )
    else:
        raise ValueError(
            f"incorrect negative log likelihood type, found {nll_type} but expected Gaussian or Bernoulli"
        )
    assert reconstruction_nll.shape[1:] == x.shape
    assert reconstruction_nll.shape[0] == n_samples

    reconstruction_nll = torch.flatten(reconstruction_nll, start_dim=2,).sum(-1)
    assert reconstruction_nll.shape == (n_samples, bsize)
    if torch.any(torch.isinf(reconstruction_nll)):
        print("warning: infinite value in reconstruction_nll")

    log_likelihood_estimates = torch.logsumexp(
        approx_posterior_nll - reconstruction_nll - prior_nll, dim=0, keepdim=False,
    ) - math.log(n_samples)
    assert log_likelihood_estimates.shape == (bsize,)

    if torch.any(torch.isinf(log_likelihood_estimates)):
        print("warning: infinite value in log likelihood estimates")

        return float("inf")
    else:
        return -log_likelihood_estimates.mean().item() / bit_per_dim_normalizer
