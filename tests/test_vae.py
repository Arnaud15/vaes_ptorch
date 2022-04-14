from typing import Tuple

import torch

import vaes_ptorch.proba as proba
import vaes_ptorch.vae as vae_nn


def smart_encoder(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mask_pos = x > 0.0
    return proba.NormalParams(
        mu=1.0 * mask_pos.double() - 1.0 * (1.0 - mask_pos.double()),
        var=torch.ones_like(x),
    )


def dumb_encoder(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mask_pos = torch.rand_like(x) > 0.5
    return proba.NormalParams(
        mu=1.0 * mask_pos.double() - 1.0 * (1.0 - mask_pos.double()),
        var=torch.ones_like(x),
    )


def identity_decoder(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return proba.NormalParams(mu=x, var=1e-6 * torch.ones_like(x))


def test_nll_is():
    bsize = 32
    n_samples = 100
    n_tests = 10
    dim = 10
    for _ in range(n_tests):
        normal_samples = torch.randn((bsize, dim))
        rd_mask = torch.rand_like(normal_samples) > 0.5
        x_data = (
            normal_samples + 1.0 * rd_mask.double() - 1.0 * (1.0 - rd_mask.double())
        )
        good_vae = vae_nn.GaussianVAE(
            encoder=smart_encoder,
            decoder=identity_decoder,
            stats_model=proba.GaussianModel(),
            latent_dim=dim,
        )
        bad_vae = vae_nn.GaussianVAE(
            encoder=dumb_encoder,
            decoder=identity_decoder,
            stats_model=proba.GaussianModel(),
            latent_dim=dim,
        )
        good_nll = good_vae.nll_is(x=x_data, n_samples=n_samples,)
        bad_nll = bad_vae.nll_is(x=x_data, n_samples=n_samples,)
        assert good_nll < bad_nll, (good_nll, bad_nll)
        print(good_nll, bad_nll)
