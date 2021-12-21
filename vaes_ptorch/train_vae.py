from typing import Optional

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .args import TrainArgs
from .losses import elbo_loss, info_vae_loss
from .utils import update_running
from .vae import GaussianVAE


def train(
    train_data: DataLoader,
    vae: GaussianVAE,
    optimizer: Optimizer,
    args: TrainArgs,
    eval_data: Optional[DataLoader] = None,
    device: str = "cpu",
):
    """Bare bones VAE training loop"""
    step = 0
    smooth_loss = None
    for epoch_ix in range(args.num_epochs):
        vae.train()
        for x in train_data:
            div_scale = args.div_annealing.get_div_scale()  # type: ignore
            x = x[0].to(device)
            optimizer.zero_grad()
            if args.info_vae:
                loss, debug_info = info_vae_loss(x, vae(x), scale=div_scale)
            else:
                loss, debug_info = elbo_loss(x, vae(x), scale=div_scale)
            loss.backward()
            optimizer.step()

            smooth_loss = update_running(
                smooth_loss, loss.item(), alpha=args.smoothing
            )
            if args.print_every and step % args.print_every == 0:
                print(
                    f"Step: {step} | Loss: {smooth_loss:.5f} | Div scale: {div_scale:.3f}"
                )
                if debug_info is not None:
                    print(debug_info)
            if args.call_every and step % args.call_every == 0:
                args.callback(vae, x, step)  # type: ignore

            step += 1

        args.div_annealing.step()  # type: ignore

        if args.eval_every and epoch_ix % args.eval_every == 0:
            assert eval_data is not None
            eval_elbo = evaluate(eval_data, vae, device=device)
            print(
                f"ELBO at the end of epoch #{epoch_ix + 1} is {eval_elbo:.5f}"
            )


def evaluate(data: DataLoader, vae: GaussianVAE, device: str = "cpu"):
    """Evaluate the ELBO of a Gaussian VAE on unseen validation data."""
    step = 0
    total_loss = 0.0
    vae.eval()
    with torch.no_grad():
        for x in data:
            x = x[0].to(device)
            loss, _ = elbo_loss(x, vae(x), scale=0.0)
            total_loss += loss.item()
            step += 1
    eval_nll = total_loss / max(step, 1)
    return eval_nll
