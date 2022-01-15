"""Helper functions for common probability operations."""

from torch import Tensor
import torch

from .models import NormalParams


def nll_is(x: torch.Tensor, vae_nn: GaussianVAE, n_samples: int = 100,) -> float:
    """Estimate the negative log likelihood of a VAE on a batch of datapoints
    `x` using importance sampling."""
    bsize = x.size(0)
    x_dims = x.shape[1:]
    dims_prod = torch.prod(torch.tensor(list(x_dims))).item()
    bit_per_dim_normalizer = math.log(2.0) * dims_prod

    z_posterior = vae_nn.encode(x)
    z_samples = vae_nn.sample_posterior(
        x, n_samples=n_samples
    )  # TODO avoid the 2 passes

    # TODO
    approx_posterior_nll = gaussian_nll(obs=z_samples, mean=mu_z, var=sig_z,).sum(-1)
    if torch.any(torch.isinf(approx_posterior_nll)):
        print("warning: infinite value in posterior")

    prior_nll = gaussian_nll(
        mean=torch.zeros_like(z_samples), obs=z_samples, var=torch.ones_like(z_samples),
    ).sum(-1)
    if torch.any(torch.isinf(prior_nll)):
        print("warning: infinite value in prior")

    x_params = vae_nn.decode(torch.flatten(z_samples, start_dim=0, end_dim=1))
    x_target = torch.tile(x.unsqueeze(0), tuple([n_samples] + [1] * x.dim()))
    reconstruction_nll = vae_nn.reconstruction_loss(x_target, dist_params=x_params)

    if torch.any(torch.isinf(reconstruction_nll)):
        print("warning: infinite value in reconstruction_nll")

    log_likelihood_estimates = torch.logsumexp(
        approx_posterior_nll - reconstruction_nll - prior_nll, dim=0, keepdim=False,
    ) - math.log(n_samples)
    assert log_likelihood_estimates.shape == (bsize,)

    if torch.any(torch.isinf(log_likelihood_estimates)):
        print("warning: infinite value in log likelihood estimates")

    return -log_likelihood_estimates.mean().item() / bit_per_dim_normalizer


def sample_bernoulli_with_logits(logits: Tensor) -> Tensor:
    """Draw independent bernoulli samples from a vector probabilities. Actually
    assumes _unnormalized_ logits in input, which must be passed through a
    sigmoid to obtain probabilities.

    Input: logits of shape (*, d)

    Output: bernoulli samples of shape (n_samples, *, d)
    """
    unif_samples = torch.rand(tuple([n_samples, *logits.size()]), device=logits.device)
    return 1.0 * (unif_samples <= torch.sigmoid(logits))


def sample_gaussian(params: NormalParams, n_samples: int = 1) -> Tensor:
    """Draw samples from a multivariate gaussian distribution.

    Input parameters:
    - mu: (*, d)
    - var: (*, d) (assumes independent components and hence a diagonal
      covariance matrix represented as a 1D tensor)

    Output:
    - samples: (n_samples, *, d)
    """
    assert params.mu.size() == params.var.size(), (params.mu.size(), params.var.size())
    assert n_samples > 0
    return mu + torch.sqrt(var) * torch.randn(
        tuple([n_samples, *mu.size()]), device=mu.device
    )


def gaussian_kl(
    left_var: Tensor, left_mu: Tensor, right_var: Tensor, right_mu: Tensor
) -> Tensor:
    """KL divergence between two multivariate gaussian distributions with
    independent components."""
    assert left_var.size() == right_var.size()
    assert left_var.size() == left_mu.size()
    assert right_var.size() == right_mu.size()

    k = left_var.shape[-1]
    trace = torch.sum(left_var / right_var, 1)
    mean_shift = torch.sum((left_mu - right_mu) ** 2 / right_var)
    log_det = torch.sum(torch.log(right_var), 1) - torch.sum(torch.log(left_var), 1)
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
