"""Functions implementing common probability operations."""
import abc
from collections import namedtuple
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

import vaes_ptorch.args as args

NormalParams = namedtuple("NormalParams", ["mu", "var",])


class StatsModel(abc.ABC):
    """Statistical Models for VAE decoders that map latent variables `z` to a
    probability distribution over the observation space `x`."""

    @staticmethod
    @abc.abstractmethod
    def compute_nll(obs: Tensor, params: Any) -> Tensor:
        """Computes the negative log likelihood of observations according to the current statistical model."""

        pass

    @staticmethod
    @abc.abstractmethod
    def sample_obs(params: Any, n_samples: int) -> Tensor:
        """Sample observations from the statistical model given current
        parameters."""
        pass


class GaussianModel(StatsModel):
    """Gaussian observation model with independent components"""

    @staticmethod
    def compute_nll(obs: Tensor, params: Any) -> Tensor:
        """Gaussian NLL"""
        return nn.GaussianNLLLoss(reduction="none", eps=args.MIN_VAR)(
            params.mu, target=obs, var=params.var
        )

    @staticmethod
    def sample_obs(params: Any, n_samples: int) -> Tensor:
        """Sample using the reparameterization trick"""
        return sample_gaussian(params=params, n_samples=n_samples)


class BernoulliModel(StatsModel):
    """Bernoulli observation model with independent components"""

    @staticmethod
    def compute_nll(obs: Tensor, params: Any) -> Tensor:
        """Binary cross entropy for _binary_ observation data (in {0, 1}),
        assuming logits rather than normalized probabilities to parameterized
        the independent bernoulli random variables"""
        return nn.BCEWithLogitsLoss(reduction="none")(params, target=obs)

    @staticmethod
    def sample_obs(params: Any, n_samples: int) -> Tensor:
        """Sampling independent bernoullis assuming logits in input"""
        return sample_bernoulli_with_logits(logits=params, n_samples=n_samples)


def sample_bernoulli_with_logits(logits: Tensor, n_samples: int) -> Tensor:
    """Draw independent bernoulli samples from a vector probability logits.
    Assumes _unnormalized_ logits in input, which must be passed through a
    sigmoid to obtain true probabilities.

    Input: logits of shape (*)

    Output: bernoulli samples of shape
    - (n_samples, *) if n_samples > 1
    - (*) otherwise
    """
    assert n_samples > 0
    unif_samples = torch.rand(tuple([n_samples, *logits.size()]), device=logits.device)
    bernoulli_samples = 1.0 * (unif_samples <= torch.sigmoid(logits))
    if n_samples == 1:
        return bernoulli_samples.squeeze(0)
    else:
        return bernoulli_samples


def sample_gaussian(params: NormalParams, n_samples: int = 1) -> Tensor:
    """Draw samples from a multivariate gaussian distribution with independent
    components.

    Input parameters:
    - mu: (*, d)
    - var: (*, d) (assumes independent components and hence a diagonal
      covariance matrix represented as a 1D tensor)

    Output samples:
    - (n_samples, *, d) if n_samples > 1
    - (*, d) otherwise
    """
    assert params.mu.size() == params.var.size(), (params.mu.size(), params.var.size())
    assert n_samples > 0
    gaussian_samples = params.mu + torch.sqrt(params.var) * torch.randn(
        tuple([n_samples, *params.mu.size()]), device=params.mu.device
    )
    if n_samples == 1:
        return gaussian_samples.squeeze(0)
    else:
        return gaussian_samples


def gaussian_kl(
    left_var: Tensor, left_mu: Tensor, right_var: Tensor, right_mu: Tensor
) -> Tensor:
    """KL divergence between two multivariate gaussian distributions with
    independent components.

    Input parameters all of shape:
    - (*, d) with `d` the dimension of the multivariate gaussian


    Output of shape (,)"""
    assert left_var.size() == right_var.size()
    assert left_var.size() == left_mu.size()
    assert right_var.size() == right_mu.size()

    k = left_var.shape[-1]
    trace = torch.sum(left_var / right_var, 1)
    mean_shift = torch.sum((left_mu - right_mu) ** 2 / right_var, 1)
    log_det = torch.sum(torch.log(right_var), 1) - torch.sum(torch.log(left_var), 1)
    return 0.5 * torch.mean(trace + mean_shift - k + log_det)


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
