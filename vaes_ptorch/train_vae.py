from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .args import TrainArgs
from .utils import update_running
from .vae import VAE


def train(data: DataLoader, vae: VAE, optimizer: Optimizer, args: TrainArgs):
    """Bare bones VAE training loop"""
    step = 0
    smooth_loss = None
    vae.train()
    for epoch_ix in range(1, args.num_epochs + 1):
        for x in data:
            optimizer.zero_grad()
            loss, debug_info = vae.elbo(x)
            loss.backward()
            optimizer.step()
            step += 1

            smooth_loss = update_running(smooth_loss, loss.item(), alpha=args.smoothing)
            if args.print_every and args.print_every % step == 0:
                print(f"Step: {step} | ELBO: {smooth_loss:.5f}")
                if debug_info is not None:
                    print(debug_info)
    return vae
