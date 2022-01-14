from typing import Tuple

import torch

import vaes_ptorch.losses as losses
import vaes_ptorch.vae as vae_nns


def smart_encoder(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mask_pos = x > 0.0
    return (
        1.0 * mask_pos.double() - 1.0 * (1.0 - mask_pos.double()),
        torch.ones_like(x),
    )


def dumb_encoder(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mask_pos = torch.rand_like(x) > 0.5
    return (
        1.0 * mask_pos.double() - 1.0 * (1.0 - mask_pos.double()),
        torch.ones_like(x),
    )


def identity_decoder(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return x, 1e-6 * torch.ones_like(x)


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
        good_vae = vae_nns.GaussianVAE(encoder=smart_encoder, decoder=identity_decoder)
        bad_vae = vae_nns.GaussianVAE(encoder=dumb_encoder, decoder=identity_decoder)
        good_nll = losses.nll_is(
            x=x_data,
            vae_nn=good_vae,
            n_samples=n_samples,
            nll_type=losses.Likelihood.Gaussian,
        )
        bad_nll = losses.nll_is(
            x=x_data,
            vae_nn=bad_vae,
            n_samples=n_samples,
            nll_type=losses.Likelihood.Gaussian,
        )
        assert good_nll < bad_nll, (good_nll, bad_nll)
        print(good_nll, bad_nll)
