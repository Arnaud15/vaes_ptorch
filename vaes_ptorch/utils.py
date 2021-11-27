from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


def sample_gaussian(mu: Tensor, var: Tensor) -> Tensor:
    """Draw samples from a multivariate gaussian distribution.

    Assumes independent components and hence a "flat" `var` parameter."""
    assert mu.size() == var.size()
    return mu + torch.sqrt(var) * torch.randn_like(mu)


def update_running(curr: Optional[float], obs: float, alpha: float) -> float:
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
