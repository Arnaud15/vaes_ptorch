from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from utils import gaussian_kl
from vae import VaeOutput


def elbo(x: Tensor, vae_output: VaeOutput) -> Tuple[Tensor, str]:
    nll = nn.GaussianNLLLoss(reduction="mean")(
        x, target=vae_output.mu_x, var=vae_output.sig_x
    )
    kl_div = gaussian_kl(
        left_mu=vae_output.mu_z,
        left_sig=vae_output.sig_z,
        right_mu=torch.zeros_like(vae_output.mu_z),
        right_sig=torch.ones_like(vae_output.sig_z),
    )
    return nll + kl_div, f"NLL: {nll.item():.5f} | KL: {kl_div.item():.5f}"
