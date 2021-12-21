from collections import namedtuple

import torch.nn as nn

from .models import GaussianModel
from .utils import sample_gaussian

VaeOutput = namedtuple("VaeOutput", ["mu_x", "sig_x", "mu_z", "sig_z"])


class GaussianVAE(nn.Module):
    def __init__(self, encoder: GaussianModel, decoder: GaussianModel):
        super(GaussianVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        mu_z, sig_z = self.encoder(x)
        posterior_z = sample_gaussian(mu=mu_z, var=sig_z)
        mu_x, sig_x = self.decoder(posterior_z)
        return VaeOutput(mu_x=mu_x, sig_x=sig_x, mu_z=mu_z, sig_z=sig_z)
