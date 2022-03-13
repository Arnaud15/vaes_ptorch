from typing import Dict, Optional, Tuple

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import vaes_ptorch.annealing as anl
import vaes_ptorch.args as args
import vaes_ptorch.utils as ut
import vaes_ptorch.vae as vae_nn


def train(
    train_data: DataLoader,
    vae: vae_nn.GaussianVAE,
    optimizer: Optimizer,
    train_args: args.TrainArgs,
    eval_data: Optional[DataLoader] = None,
    device: str = "cpu",
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Training loop with periodic logging."""
    step = 1
    train_info = {}
    eval_info = {}
    divergence = vae_nn.Divergence.MMD if train_args.info_vae else vae_nn.Divergence.KL
    prev_div_value = None
    div_annealing: anl.AnnealingSchedule = train_args.div_annealing  # type: ignore
    for epoch_ix in range(1, train_args.num_epochs + 1):
        for x in train_data:
            div_scale = div_annealing.get_scale(prev_div_value)
            (loss, loss_info) = vae.loss(
                x.to(device), div_type=divergence, div_scale=div_scale
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prev_div_value = loss_info["div"]
            loss_info["div_scale"] = div_scale
            ut.update_info_dict(train_info, loss_info)
            if train_args.print_every and step % train_args.print_every == 0:
                print(
                    f"Training logs for step: {step}, epoch: {epoch_ix}, info: {ut.print_info_dict(train_info)}"
                )

            if train_args.eval_every and step % train_args.eval_every == 0:
                assert eval_data is not None
                eval_results = evaluate(eval_data, vae, device=device)
                vae.train()
                ut.update_info_dict(eval_info, eval_results)
                print(f"Validation logs for step: {step}, epoch: {epoch_ix}")

            step += 1

        train_args.div_annealing.step()  # type: ignore
    return train_info, eval_info


def evaluate(
    eval_data: DataLoader,
    vae: vae_nn.GaussianVAE,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluate the ELBO and estimated NLL of a Gaussian VAE on a dataset."""
    step = 0
    total_nll = 0.0
    total_elbo = 0.0
    vae.eval()
    with torch.no_grad():
        for x in eval_data:
            x = x.to(device)
            total_nll += vae.nll_is(x)
            total_elbo += vae.loss(x, div_type=vae_nn.Divergence.KL, div_scale=1.0)[1][
                "loss"
            ]
            step += 1
    eval_nll = total_nll / max(step, 1)
    eval_elbo = total_elbo / max(step, 1)
    return {"elbo": eval_elbo, "nll": eval_nll}
