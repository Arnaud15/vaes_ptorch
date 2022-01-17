from collections import namedtuple
from typing import Optional, Tuple

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import vaes_ptorch.args as args
import vaes_ptorch.utils as ut
import vaes_ptorch.vae as vae_nn

Results = namedtuple("Results", ["train_ewma", "eval_ewma"])


def train(
    train_data: DataLoader,
    vae: vae_nn.GaussianVAE,
    optimizer: Optimizer,
    train_args: args.TrainArgs,
    eval_data: Optional[DataLoader] = None,
    device: str = "cpu",
) -> Results:
    """Bare bones VAE training loop"""
    step = 0
    train_loss = None
    eval_loss = None
    divergence = vae_nn.Divergence.MMD if train_args.info_vae else vae_nn.Divergence.KL
    for epoch_ix in range(train_args.num_epochs):
        vae.train()
        for x in train_data:
            div_scale = train_args.div_annealing.get_div_scale()  # type: ignore
            x = x[0].to(device)
            optimizer.zero_grad()
            elbo = vae.compute_elbo(x, div_type=divergence, div_scale=div_scale)
            elbo.loss.backward()
            optimizer.step()

            train_loss = ut.update_running(
                train_loss, elbo.loss.item(), alpha=train_args.smoothing
            )
            if train_args.print_every and step % train_args.print_every == 0:
                print(
                    f"Step: {step} | Training loss: {train_loss:.5f} | Div scale: {div_scale:.3f}"
                )
                print(f"NLL: {elbo.nll:.5f} | {divergence}: {elbo.div:.5f}")

            step += 1

        train_args.div_annealing.step()  # type: ignore

        if train_args.eval_every and epoch_ix % train_args.eval_every == 0:
            assert eval_data is not None
            eval_elbo = evaluate(eval_data, vae, train_args=train_args, device=device)
            print(f"ELBO at the end of epoch #{epoch_ix + 1} is {eval_elbo:.5f}")
            eval_loss = ut.update_running(
                eval_loss, eval_elbo, alpha=train_args.smoothing
            )
    return Results(train_ewma=train_loss, eval_ewma=eval_loss)


def evaluate(
    data: DataLoader,
    vae: vae_nn.GaussianVAE,
    train_args: args.TrainArgs,
    device: str = "cpu",
) -> float:
    """Evaluate the ELBO of a Gaussian VAE on data."""
    step = 0
    total_nll = 0.0
    vae.eval()
    with torch.no_grad():
        for x in data:
            x = x[0].to(device)
            total_nll += vae.nll_is(x)
            step += 1
    eval_nll = total_nll / max(step, 1)
    return eval_nll
