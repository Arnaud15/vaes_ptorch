from typing import Union
from collections import namedtuple

from torch import Tensor
import torch.nn as nn

from .models import NormalParams
from .utils import sample_gaussian
import vaes_ptorch.proba as proba

DistParams = Union[NormalParams, Tensor]

VaeOutput = namedtuple("VaeOutput", ["z_params", "x_params",])

ElboLoss = namedtuple("ElboLoss", ["loss", "nll", "div"])


class Likelihood(enum.Enum):
    Gaussian = enum.auto()
    Bernoulli = enum.auto()


class Divergence(enum.Enum):
    KL = enum.auto()
    MMD = enum.auto()


class GaussianVAE(nn.Module):
    """
    VAE class representing VAE models with the following constraint:
    - the prior distribution over the latent space is unit multivariate normal
    - the variational posterior over that latent space parameterizes a
      multivariate gaussian distribution with independent components
    """

    def __init__(
        self,
        encoder: GaussianModel,
        decoder: nn.Module,
        latent_dim: int,
        likelihood: Likelihood,
    ):
        super(GaussianVAE, self).__init__()
        assert latent_dim > 0, f"found nonpositive latent dim: {latent_dim}"
        self.latent_dim = latent_dim
        self.likelihood = likelihood
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x: Tensor) -> NormalParams:
        return self.encoder(x)

    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)

    def sample_posterior(self, x: Tensor, n_samples: int = 1) -> Tensor:
        z_params = self.encode(x)
        return proba.sample_gaussian(z_params, n_samples=n_samples)

    def sample_prior(self, n_samples: int, device: str) -> Tensor:
        assert (
            n_samples > 0
        ), f"found nonpositive number of samples to generate: {n_samples}"
        return proba.sample_gaussian(
            NormalParams(
                mu=torch.zeros(self.latent_dim, device=device),
                var=torch.ones(self.latent_dim, device=device),
            ),
            n_samples=n_samples,
        )

    def sample_obs(self, dist_params: DistParams, n_samples: int) -> Tensor:
        if self.likelihood == Likelihood.Gaussian:
            return proba.sample_gaussian(dist_params)
        elif self.likelihood == Likelihood.Bernoulli:
            return proba.sample_bernoulli_with_logits(dist_params)
        else:
            raise ValueError("unexpected likelihood value: {self.likelihood}")

    def generate_x(self, n_samples: int, device: str) -> Tensor:
        assert (
            n_samples > 0
        ), f"found nonpositive number of samples to generate: {n_samples}"
        prior_samples = self.sample_prior(n_samples, device)
        x_params = self.decode(prior_samples)
        return self.sample_obs(x_params, n_samples)

    def forward(self, x: Tensor) -> VaeOutput:
        z_sample = self.sample_posterior(x)
        x_params = self.decode(z_sample)
        return VaeOutput(z_sample=z_sample, x_params=x_params)

    def reconstruction_loss(self, x: Tensor, dist_params: DistParams) -> Tensor:
        batch_size = x.size(0)
        if self.likelihood == Likelihood.Gaussian:
            return (
                nn.GaussianNLLLoss(reduction="sum", eps=1.0)(
                    dist_params.mu, target=x, var=dist_params.var
                )
                / batch_size
            )
        elif self.likelihood == Likelihood.Bernoulli:
            return (
                nn.BCEWithLogitsLoss(reduction="sum")(dist_params, target=x)
                / batch_size
            )
        else:
            raise ValueError("unexpected likelihood value: {self.likelihood}")

    def divergence_loss(self, z_params: NormalParams, div_type: Divergence) -> Tensor:
        if div_type == Divergence.KL:
            return proba.gaussian_kl(
                left_mu=z_params.mu,
                left_var=z_params.var,
                right_mu=torch.zeros_like(z_params.mu),
                right_var=torch.ones_like(z_params.var),
            )
        elif div_type == Divergence.MMD:
            # z samples from the variational posterior distribution
            # z ~ p_data(x) * q_{phi}(z | x)
            z_samples = proba.sample_gaussian(z_params, n_samples=1).squeeze(
                0
            )  # z samples from the prior distribution
            z_prior = self.sample_prior(
                n_samples=z_samples.size(0), device=z_samples.device
            )
            return proba.mmd_rbf(samples_p=z_samples, samples_q=z_prior,)
        else:
            raise ValueError(
                f"unrecognized divergence type: {div_type}, expected [KL | MMD]"
            )

    def compute_elbo(
        self, x: Tensor, div_type: Divergence, div_scale: float = 1.0
    ) -> ElboLoss:
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
        output = self.forward(x)
        reconstruction_loss = self.reconstruction_loss(output.x_params, x)
        div_term = self.divergence_loss(output.z_params)
        return ElboLoss(
            loss=reconstruction_loss + div_scale * div_term,
            nll=reconstruction_loss.item(),
            div=div_term.item(),
        )
