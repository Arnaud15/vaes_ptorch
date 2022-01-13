from typing import Optional

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F  # type: ignore
from torch import Tensor
from torchvision.utils import make_grid  # type: ignore


def gaussian_nll(obs: Tensor, mean: Tensor, var: Tensor, eps: float = 1e-6) -> Tensor:
    return (obs - mean) ** 2 / (var + eps) + torch.log(var + eps)


def sample_gaussian(mu: Tensor, var: Tensor, n_samples: int = 1) -> Tensor:
    """Draw samples from a multivariate gaussian distribution.

    Assumes independent components and hence a "flat" `var` parameter."""
    assert mu.size() == var.size(), (mu.size(), var.size())
    assert n_samples > 0

    if n_samples == 1:
        return mu + torch.sqrt(var) * torch.randn_like(mu)
    else:
        return mu + torch.sqrt(var) * torch.randn(
            tuple([n_samples, *mu.size()]), device=mu.device
        )


def update_running(curr: Optional[float], obs: float, alpha: float) -> float:
    """Update an exponentially weighted moving average with a new observation.
    
    If the current value of the moving average has not been initialized already
    it is `None` and set equal to the new observation."""

    assert alpha >= 0.0 and alpha < 1.0

    if curr is None:
        return obs
    else:
        return obs * (1.0 - alpha) + curr * alpha


def gaussian_kl(
    left_sig: Tensor, left_mu: Tensor, right_sig: Tensor, right_mu: Tensor
) -> Tensor:
    """KL divergence between two multivariate gaussian distributions with
    independent components."""
    assert left_sig.size() == right_sig.size()
    assert left_sig.size() == left_mu.size()
    assert right_sig.size() == right_mu.size()
    assert left_sig.dim() == 2

    k = left_sig.size(1)
    trace = torch.sum(left_sig / right_sig, 1)
    mean_shift = torch.sum((left_mu - right_mu) ** 2 / right_sig)
    log_det = torch.sum(torch.log(right_sig), 1) - torch.sum(torch.log(left_sig), 1)
    return 0.5 * torch.mean(trace + mean_shift - k + log_det)


def rbf_kernel(left_samples: Tensor, right_samples: Tensor, bandwidth: float) -> Tensor:
    """Computes the mean RBF kernel between to sets of samples in the same
    euclidian space.

    Effectively computes mean over samples: exp ( - 1/2 * ||left - right|| ^ 2
    / bandwidth ).

    Each left sample is compared to each right sample, so the average is over
    n_left * n_right kernel distances."""
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


def show(img: torch.Tensor):
    """Small utility to plot a tensor of images"""
    img = img.detach()
    try:
        img = F.to_pil_image(img)
    except ValueError:  # handle batched images to plot
        img = make_grid(img)
        img = F.to_pil_image(img)
    plt.imshow(np.asarray(img))
    plt.xticks([])  # remove pyplot borders
    plt.yticks([])
    plt.show()
    plt.close()
