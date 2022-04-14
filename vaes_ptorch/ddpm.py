import enum
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class SigmaSchedule(enum.Enum):
    Beta = enum.auto()
    BetaTilde = enum.auto()


def make_linear_beta(n_timesteps: int) -> Tensor:
    """Linear increasing beta from the paper."""
    return torch.linspace(1e-4, 0.02, n_timesteps)


def make_sigma2(choice: SigmaSchedule, beta: Tensor, alpha_hat: Tensor) -> Tensor:
    """Choice of the reconstruction process variances."""
    if choice == SigmaSchedule.Beta:
        return beta
    if choice == SigmaSchedule.BetaTilde:
        alpha_hat_m1 = torch.cat([torch.ones(1), alpha_hat[:-1]])
        assert alpha_hat_m1.shape == alpha_hat.shape
        return beta * (1.0 - alpha_hat_m1) / (1 - alpha_hat)


def make_alpha_hat(beta: Tensor) -> Tensor:
    """Return the product of alphas."""
    return torch.cumprod(1 - beta, 0)


class DDPM(nn.Module):
    def __init__(
        self,
        net: nn.Module,
        n_timesteps: int,
        sigma: SigmaSchedule = SigmaSchedule.BetaTilde,
    ):
        super().__init__()
        assert n_timesteps > 0
        self.n_timesteps = n_timesteps
        self.register_buffer("beta", make_linear_beta(n_timesteps))
        self.register_buffer("alpha_hat", make_alpha_hat(self.beta))
        self.register_buffer(
            "sigma2", make_sigma2(sigma, beta=self.beta, alpha_hat=self.alpha_hat)
        )
        self.net = net

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        bsize = x.shape[0]
        rd_t_index = torch.randint(
            high=self.beta.shape[0], size=(bsize,), device=x.device
        )
        alpha_hat_t = self.alpha_hat[rd_t_index].unsqueeze(-1)
        # Noisify input
        noise = torch.randn_like(x)
        x_t = alpha_hat_t.sqrt() * x + (1 - alpha_hat_t).sqrt() * noise
        # Denoising network and loss
        pred_noise = self.net(x_t, 1 + rd_t_index.unsqueeze(-1))
        assert pred_noise.shape == noise.shape
        return pred_noise, noise

    def sample(self, device: torch.device, shape: torch.Size,) -> Tensor:
        assert len(shape)
        x_t = torch.randn(shape, device=device)
        zero = torch.zeros((shape[0], 1), dtype=torch.long, device=device)
        for t in range(self.beta.shape[0], 0, -1):
            beta_t, alpha_hat_t, sigma2_t = (
                self.beta[t - 1],
                self.alpha_hat[t - 1],
                self.sigma2[t - 1],
            )
            noise_pred = self.net(x_t, zero + t)
            mu_theta_t = (1.0 - beta_t).rsqrt() * (
                x_t - beta_t * (1.0 - alpha_hat_t).rsqrt() * noise_pred
            )
            # zero noise for last sampling step
            eps = 0.0 if t == 1 else torch.randn_like(mu_theta_t) * sigma2_t.sqrt()
            x_t = eps + mu_theta_t
        return x_t
