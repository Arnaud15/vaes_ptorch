"""Utilities around probability distributions and operations."""
import enum
from typing import Any, Optional

import torch
import torch.distributions as tdist
from torch import Tensor
from torch.distributions.distribution import Distribution

import vaes_ptorch.args as args


class ObsModel(enum.Enum):
    """Supported observation models for VAEs"""

    Bernoulli = enum.auto()
    Gaussian = enum.auto()
    UnitGaussian = enum.auto()


def to_gaussian_dist(
    params: Tensor, dim: int, min_std: float = args.MIN_STD
) -> tdist.normal.Normal:
    assert (
        params.shape[-1] == 2 * dim
    ), f"unexpected output dimension, expected: {dim * 2} for posterior mean + std, found: {params.shape}"
    mean, log_std = torch.split(params, split_size_or_sections=dim, dim=-1)
    return tdist.normal.Normal(loc=mean, scale=torch.exp(0.5 * log_std) + min_std)


def params_to_dist(params: Any, obs_model: ObsModel) -> Distribution:
    """Map a supported VAE observation model and its parameters to the
    corresponding torch Distribution object."""
    if obs_model == ObsModel.Bernoulli:
        # expecting unnormalized logits in input
        return tdist.bernoulli.Bernoulli(logits=params)
    elif obs_model == ObsModel.UnitGaussian:
        # unit variance multivariate normal w/ indep components
        return tdist.normal.Normal(loc=params, scale=1.0)
    elif obs_model == ObsModel.Gaussian:
        # multivariate normal w/ indep components and diag covariance
        return to_gaussian_dist(params, params.shape[-1] // 2)
    else:
        raise NotImplementedError("unsupported obs model")


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
