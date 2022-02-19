"""Functions implementing common probability operations."""
from typing import Optional

import torch
from torch import Tensor


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
        proba.GaussianModel().compute_nll(obs=z_samples, params=z_params_dup).sum(-1)
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

    x_params = self.decoder(torch.flatten(z_samples, start_dim=0, end_dim=1))
    x_target = torch.tile(x.unsqueeze(0), tuple([n_samples] + [1] * x.dim())).reshape(
        n_samples * bsize, *x_dims
    )
    reconstruction_nll = self.obs_model.compute_nll(obs=x_target, params=x_params)
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


def rbf_kernel(left_samples: Tensor, right_samples: Tensor, bandwidth: float) -> Tensor:
    """Computes the mean RBF kernel between to sets of samples associated to
    distributions (P, Q) whose outputs live in the same euclidian space of
    dimension `d`.

    Effectively computes the mean over samples of:
    exp ( - 1/2 * ||left - right|| ^ 2 / bandwidth )

    Each left sample is compared to each right sample, so the average is over
    n_left * n_right kernel distances.

    Input samples of shape (n_left, d), (n_right, d)

    Output of shape (,)"""
    assert bandwidth > 0.0
    assert left_samples.dim() == 2 and right_samples.dim() == 2
    assert left_samples.size(1) == right_samples.size(1)

    left_size = left_samples.size(0)
    right_size = right_samples.size(0)
    left_tiled = torch.tile(left_samples, (right_size, 1))
    right_tiled = torch.repeat_interleave(right_samples, left_size, 0)

    assert (
        left_tiled.size(0) == right_tiled.size(0)
        and left_tiled.size(0) == left_size * right_size
    )
    assert left_tiled.dim() == right_tiled.dim() and left_tiled.dim() == 2

    square_dists = ((left_tiled - right_tiled) ** 2).sum(1)

    assert square_dists.dim() == 1
    assert square_dists.size(0) == left_size * right_size

    return torch.mean(torch.exp(-0.5 * square_dists / bandwidth))


def mmd_rbf(samples_p, samples_q, bandwidth: Optional[float] = None) -> Tensor:
    """Computes and returns a finite-sample estimate of the Maximum Mean
    Discrepancy (MMD) between two distributions (P, Q) from samples of P and Q
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
