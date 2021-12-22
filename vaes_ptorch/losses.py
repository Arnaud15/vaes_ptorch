import enum
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .utils import gaussian_kl, rbf_kernel
from .vae import VaeOutput


class Nll(enum.Enum):
    Gaussian = enum.auto()
    Bernoulli = enum.auto()


def elbo_loss(
    x: Tensor, vae_output: VaeOutput, nll_type: Nll, scale: float = 1.0
) -> Tuple[Tensor, str]:
    """Computes the ELBO (Evidence Lower Bound) loss and returns the loss along
    with debug information about the different loss components.

    Assumes that
    - The vae decoder parameterizes a multivariate gaussian distribution with
      independent components.
    - The prior distribution over latents is a vector of independent
    unit gaussian distributions.

    Allows the re-scaling of the KL divergence via the `scale` parameter
    (useful to implement Beta VAE and stabilize training more generally).


    From the original VAE paper: https://arxiv.org/abs/1312.6114.
    """
    assert scale >= 0.0
    assert x.dim() > 1  # assume the first dimension is the batch
    batch_size = x.size(0)
    assert batch_size > 0

    if nll_type == Nll.Gaussian:
        nll = nn.GaussianNLLLoss(reduction="sum", eps=1.0)(
            vae_output.mu_x, target=x, var=vae_output.sig_x
        )
    elif nll_type == Nll.Bernoulli:
        nll = nn.BCEWithLogitsLoss(reduction="sum")(vae_output.mu_x, target=x)
    else:
        raise ValueError(
            f"incorrect negative log likelihood type, found {nll_type} but expected Gaussian or Bernoulli"
        )
    nll /= batch_size
    assert not nll.dim(), nll.dim()

    kl_div = gaussian_kl(
        left_mu=vae_output.mu_z,
        left_sig=vae_output.sig_z,
        right_mu=torch.zeros_like(vae_output.mu_z),
        right_sig=torch.ones_like(vae_output.sig_z),
    )
    return nll + scale * kl_div, f"NLL: {nll.item():.5f} | KL: {kl_div.item():.5f}"


# TODO -> merge with the above elbo, only the div differs...
def info_vae_loss(
    x: Tensor, vae_output: VaeOutput, nll_type: Nll, scale: float = 1.0
) -> Tuple[Tensor, str]:
    """Computes the InfoVAE loss.

    Compared to the original InfoVAE paper (https://arxiv.org/abs/1706.02262),
    we restrain the loss to the setting `alpha` = 1. The way our loss is
    parameterized, the `scale` argument is exactly lambda from the paper. We
    also only support the MMD divergence.

    This setting is exactly that of MMD-VAE:
    https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/.
    """
    assert scale >= 0.0
    assert x.dim() > 1  # assume the first dimension is the batch
    batch_size = x.size(0)
    assert batch_size > 0

    if nll_type == Nll.Gaussian:
        nll = nn.GaussianNLLLoss(reduction="sum", eps=1.0)(
            vae_output.mu_x, target=x, var=vae_output.sig_x
        )
    elif nll_type == Nll.Bernoulli:
        nll = nn.BCEWithLogitsLoss(reduction="sum")(vae_output.mu_x, target=x)
    else:
        raise ValueError(
            f"incorrect negative log likelihood type, found {nll_type} but expected Gaussian or Bernoulli"
        )
    nll /= batch_size
    assert not nll.dim(), nll.dim()

    z_samples = (
        torch.randn_like(vae_output.mu_z) * torch.sqrt(vae_output.sig_z)
        + vae_output.mu_z
    )
    z_prior = torch.randn_like(vae_output.mu_z)

    mmd_div = mmd_rbf(samples_p=z_samples, samples_q=z_prior,)
    return (
        nll + scale * mmd_div,
        f"NLL: {nll.item():.5f} | MMD-div: {mmd_div.item():.5f}",
    )


def mmd_rbf(samples_p, samples_q, bandwidth: Optional[float] = None) -> Tensor:
    """Computes and returns a finite-sample estimate of the Maximum Mean
    Discrepancy (MMD) between two distributions P, Q from samples of P and Q
    and using a RBF kernel.

    If the bandwidth of the RBF is not specified, set it to dim / 2, where dim
    is the dimension of the samples."""
    assert samples_p.dim() == 2 and samples_q.dim() == 2
    assert samples_q.size(1) == samples_q.size(1)

    if bandwidth is None:
        bandwidth = samples_p.size(1) / 2.0

    kernel_p = rbf_kernel(samples_p, samples_p, bandwidth)
    kernel_q = rbf_kernel(samples_q, samples_q, bandwidth)
    kernel_pq = rbf_kernel(samples_p, samples_q, bandwidth)

    return kernel_p + kernel_q - 2 * kernel_pq
