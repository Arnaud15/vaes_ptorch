import enum
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor

import vaes_ptorch.models as models


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


class DDPMNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_timesteps: int,
        h_dim: int,
        n_hidden: int,
        fourier_inputs: bool = False,
        fourier_t: bool = False,
        fourier_dim: Optional[int] = None,
    ):
        super().__init__()
        if fourier_inputs:
            assert fourier_dim is not None
            self.proj_x = models.FourierFeatures(in_dim=in_dim, out_dim=fourier_dim, scale=1.0)
            x_dim = fourier_dim
        else:
            self.proj_x = nn.Identity()
            x_dim = in_dim

        if fourier_t:
            assert fourier_dim is not None
            self.proj_t = nn.Sequential(
                models.EmbeddingFourier(n_pos=n_timesteps, dim=fourier_dim),
                nn.Linear(fourier_dim, h_dim),
                nn.SiLU(),
                nn.Linear(h_dim, h_dim),
            )
            t_dim = h_dim
        else:
            self.proj_t = lambda x: x/n_timesteps
            t_dim = 1

        self.encode = models.get_mlp(in_dim=x_dim + t_dim, out_dim=in_dim, h_dim=h_dim, n_hidden=n_hidden)


    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        assert t.dtype == torch.long
        x = self.proj_x(x)
        t = self.proj_t(t)
        return self.encode(torch.cat([x, t], dim=-1))


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


if __name__ == "__main__":
    import numpy as np
    import vaes_ptorch.plot as plot
    import vaes_ptorch.trainer as trainer

    # data gen params
    DSET_SIZE = 16384
    DIM = 8
    BATCH_SIZE = 128

    # training params
    N_STEPS = 5_000
    PRINT_EVERY = N_STEPS // 25

    # DDPM params
    T = 1000
    H_DIM_DDPM = 32
    N_LAYERS_DDPM = 4
    LR_DDPM = 1e-3
    FOURIER_DIM = 32
    N_SAMPLES_DDPM = 1000

    # data gen
    P = torch.zeros((2, DIM))
    P[0, ::2] = 1
    P[1, 1::2] = 1
    data_x = torch.linspace(0, 2 * np.pi, 7)[:-1]
    data_x = torch.stack((torch.cos(data_x), torch.sin(data_x)), dim=1)
    data_x = data_x[None] + torch.randn((DSET_SIZE, data_x.shape[0], 2)) * 0.1
    data_x = data_x.view(-1, 2)
    plot.plot_points_series([data_x[i::6].numpy() for i in range(6)])
    def data_it(dataset, proj, batch_size):
        for _ in range(0, len(dataset), batch_size):
            idx = torch.randint(len(dataset), size=(batch_size,))
            batch = dataset[idx] @ proj
            yield batch
    train_data = data_it(data_x, P, BATCH_SIZE)

    # ddpm training
    ddpm_net = DDPMNet(
        in_dim=DIM,
        fourier_dim=FOURIER_DIM,
        h_dim=H_DIM_DDPM,
        n_hidden=N_LAYERS_DDPM,
        n_timesteps=T,
        fourier_inputs=False,
        fourier_t=True,
    )
    ddpm = DDPM(
        net=ddpm_net, n_timesteps=T, sigma=SigmaSchedule.BetaTilde
    )

    # loss and optimizer
    opt = torch.optim.Adam(ddpm.parameters(), lr=LR_DDPM)
    def mse_loss(pred_eps, true_eps):
        loss = torch.nn.functional.mse_loss(pred_eps, true_eps, reduction='mean')
        return loss, {'mse_loss': loss.item()}

    # training
    args = trainer.TrainArgs(N_STEPS, PRINT_EVERY)
    trainer.train_loop(ddpm, opt, mse_loss, args, train_data)
    
    with torch.no_grad():
        samples = ddpm.sample(device="cpu", shape=(N_SAMPLES_DDPM, DIM))
        plot.plot_points_series([samples[i::6].numpy() for i in range(6)])
