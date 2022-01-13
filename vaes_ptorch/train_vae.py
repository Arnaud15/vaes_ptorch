from collections import namedtuple
from typing import Optional, Tuple

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .args import TrainArgs
from .losses import Divergence, Likelihood, elbo_loss
from .utils import update_running
from .vae import GaussianVAE

Results = namedtuple("Results", ["train_ewma", "eval_ewma"])


def train(
    train_data: DataLoader,
    vae: GaussianVAE,
    optimizer: Optimizer,
    args: TrainArgs,
    eval_data: Optional[DataLoader] = None,
    device: str = "cpu",
) -> Results:
    """Bare bones VAE training loop"""
    step = 0
    train_loss = None
    eval_loss = None
    divergence = Divergence.MMD if args.info_vae else Divergence.KL
    for epoch_ix in range(args.num_epochs):
        vae.train()
        for x in train_data:
            div_scale = args.div_annealing.get_div_scale()  # type: ignore
            x = x[0].to(device)
            optimizer.zero_grad()
            loss, (elbo, div) = elbo_loss(
                x,
                vae(x),
                nll_type=args.likelihood,
                div_type=divergence,
                div_scale=div_scale,
            )
            loss.backward()
            optimizer.step()

            train_loss = update_running(
                train_loss, loss.item(), alpha=args.smoothing
            )
            if args.print_every and step % args.print_every == 0:
                print(
                    f"Step: {step} | Training loss: {train_loss:.5f} | Div scale: {div_scale:.3f}"
                )
                print(
                    f"{args.likelihood}: {elbo:.5f} | {divergence}: {div:.5f}"
                )

            step += 1

        args.div_annealing.step()  # type: ignore

        if args.eval_every and epoch_ix % args.eval_every == 0:
            assert eval_data is not None
            eval_elbo = evaluate(eval_data, vae, args=args, device=device)
            print(
                f"ELBO at the end of epoch #{epoch_ix + 1} is {eval_elbo:.5f}"
            )
            eval_loss = update_running(
                eval_loss, eval_elbo, alpha=args.smoothing
            )
    return Results(train_ewma=train_loss, eval_ewma=eval_loss)


def evaluate(
    data: DataLoader, vae: GaussianVAE, args: TrainArgs, device: str = "cpu"
) -> float:
    """Evaluate the ELBO of a Gaussian VAE on data."""
    step = 0
    total_loss = 0.0
    vae.eval()
    with torch.no_grad():
        for x in data:
            x = x[0].to(device)
            loss, _ = elbo_loss(
                x,
                vae(x),
                div_type=Divergence.KL,  # only the plain VAE loss with scale 1.0 is a true lower bound to the log likelihood
                nll_type=args.likelihood,
                div_scale=1.0,
            )
            total_loss += loss.item()
            step += 1
    eval_nll = total_loss / max(step, 1)
    return eval_nll
