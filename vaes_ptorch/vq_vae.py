"""The VQ-VAE class."""
from typing import Dict, Tuple

import einops as ei
import torch
import torch.nn as nn
from torch import Tensor

import vaes_ptorch.models as models

# TODO -> docstrings for function params


class VQVAE(nn.Module):
    """Simple VQ-VAE with 1D sequence of quantized latents."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int,
        codebook_size: int = 10,
        seq_len: int = 5,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vector_quantizer = models.VectorQuantizer1D(
            codebook_size=codebook_size,
            dim=latent_dim,
        )
        self.latent_dim = latent_dim
        self.seq_len = seq_len

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        z_e = self.encoder(x)
        assert z_e.shape[-1] == self.latent_dim, z_e.shape
        assert z_e.shape[-2] == self.seq_len, z_e.shape
        z_q, indices = self.vector_quantizer(z_e)
        assert z_q.shape[-1] == self.latent_dim, z_q.shape
        assert z_q.shape[-2] == self.seq_len, z_q.shape
        z_q_ste = z_e + (z_q - z_e).detach()
        x_rec = self.decoder(z_q_ste)
        assert x_rec.shape == x.shape, (x_rec.shape, x.shape)
        return (z_e, z_q, x_rec, indices)

    def sample_uniform(self, n_samples: int) -> Tensor:
        assert n_samples > 0, n_samples
        z_q = self.vector_quantizer.sample_uniform(n_samples, seq_len=self.seq_len)
        return self.decoder(z_q)

    def loss(
        self,
        x: Tensor,
        z_e: Tensor,
        z_q: Tensor,
        x_rec: Tensor,
        commitment_weight: float = 0.25,
    ) -> Tuple[Tensor, Dict[str, float]]:
        assert commitment_weight > 0.0, commitment_weight
        rec_mse = ei.reduce((x - x_rec) ** 2, "b ... -> b", "sum").mean()

        e_latent_loss = ei.reduce((z_e - z_q.detach()) ** 2, "b ... -> b", "sum").mean()

        q_latent_loss = ei.reduce((z_e.detach() - z_q) ** 2, "b ... -> b", "sum").mean()

        loss = rec_mse + e_latent_loss + commitment_weight * q_latent_loss

        return loss, {
            "rec_mse": rec_mse.item(),
            "e_latent_loss": e_latent_loss.item(),
            "q_latent_loss": q_latent_loss.item(),
        }

    def sample_autoregressive(
        self, seq_model: models.SeqModel, n_samples: int, hidden_dim: int, seq_len: int
    ) -> Tensor:
        """Sample from a VQ-VAE with 1D quantized latents using a trained 1D
        autoregressive model to sample the latent vectors."""
        h_0 = torch.zeros(1, n_samples, hidden_dim)
        samples = []
        for _ in range(seq_len):
            logits = seq_model.proj_head(ei.rearrange(h_0, "1 b h -> b h"))
            idx_sampled = torch.multinomial(
                torch.exp(logits), num_samples=1, replacement=True
            )
            z_q = self.vector_quantizer.codebook(idx_sampled)
            samples.append(z_q)
            _, h_0 = seq_model.gru(z_q, h_0)
        samples = torch.cat(samples, dim=1)
        samples = self.decoder(samples)
        return samples


if __name__ == "__main__":
    import numpy as np
    import torch.nn.functional as F

    import vaes_ptorch.plot as plot
    import vaes_ptorch.utils as ut

    # params
    DSET_SIZE = 16384
    DIM = 16

    BATCH_SIZE = 512
    LR = 1e-2
    N_EPOCHS = 10

    SEQ_LEN = 5
    VOCAB_SIZE = 10
    LATENT_DIM = 1
    N_LAYERS = 1
    H_DIM = 128
    H_DIM_GRU = 128

    # model and optimizer init
    encoder = nn.Sequential(
        models.get_mlp(
            in_dim=DIM, out_dim=LATENT_DIM * SEQ_LEN, h_dim=H_DIM, n_hidden=N_LAYERS
        ),
        nn.Unflatten(dim=1, unflattened_size=(SEQ_LEN, LATENT_DIM)),
    )
    decoder = nn.Sequential(
        nn.Flatten(start_dim=1, end_dim=-1),
        models.get_mlp(
            in_dim=SEQ_LEN * LATENT_DIM, out_dim=DIM, h_dim=H_DIM, n_hidden=N_LAYERS
        ),
    )
    net = VQVAE(
        encoder=encoder,
        decoder=decoder,
        latent_dim=LATENT_DIM,
        codebook_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
    )
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    # data generation
    P = torch.zeros((2, DIM))
    P[0, ::2] = 1
    P[1, 1::2] = 1
    data_x = torch.linspace(0, 2 * np.pi, 7)[:-1]
    data_x = torch.stack((torch.cos(data_x), torch.sin(data_x)), dim=1)
    data_x = data_x[None] + torch.randn((DSET_SIZE, data_x.shape[0], 2)) * 0.1
    data_x = data_x.view(-1, 2)
    plot.plot_points_series([data_x[i::6].numpy() for i in range(6)])

    # VQ-VAE reconstruction training
    net.train()
    for epoch in range(N_EPOCHS):

        info = {}

        for _ in range(0, DSET_SIZE, BATCH_SIZE):
            idx = torch.randint(data_x.shape[0], size=(BATCH_SIZE,))
            batch = data_x[idx] @ P
            z_e, z_q, x_reconstructed, _ = net(batch)
            loss, loss_info = net.loss(batch, z_e=z_e, z_q=z_q, x_rec=x_reconstructed)

            opt.zero_grad()
            loss.backward()
            opt.step()

            ut.update_info_dict(info, obs=loss_info)

        print(f"Epoch {epoch + 1} | Info: {ut.print_info_dict(info)}.")

    # Testing the reconstruction
    net.eval()
    with torch.no_grad():
        series = []
        for i in range(6):
            full_batch = data_x[i::6] @ P
            _, _, x_rec, _ = net(full_batch)
            series.append(x_rec.numpy())
        plot.plot_points_series(series)

    # uniform sampling
    with torch.no_grad():
        plot.plot_points_series([net.sample_uniform(DSET_SIZE).numpy()])

    seq_model = models.SeqModel(
        input_size=LATENT_DIM, hidden_size=H_DIM_GRU, codebook_size=VOCAB_SIZE
    )

    # Training an autoregressive model over the 1D VQ-VAE latent sequences
    opt = torch.optim.Adam(seq_model.parameters(), lr=LR)
    seq_model.train()
    for epoch in range(N_EPOCHS):

        tot_loss = None

        for _ in range(0, DSET_SIZE, BATCH_SIZE):
            idx = torch.randint(data_x.shape[0], size=(BATCH_SIZE,))
            batch = data_x[idx] @ P
            _, z_q, _, indices = net(batch)
            logits = seq_model(z_q.detach())
            loss = (
                F.cross_entropy(
                    ei.rearrange(logits, "n l c -> n c l"),
                    indices.detach(),
                    reduction="sum",
                )
                / BATCH_SIZE
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            tot_loss = ut.update_running(
                tot_loss,
                loss.item(),
            )

        print(f"Epoch {epoch + 1} | Loss {tot_loss:.5f}")

    # autoregressive sampling
    with torch.no_grad():
        plot.plot_points_series(
            [
                net.sample_autoregressive(
                    seq_model=seq_model,
                    n_samples=DSET_SIZE,
                    hidden_dim=H_DIM_GRU,
                    seq_len=SEQ_LEN,
                ).numpy()
            ]
        )
