from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .args import TrainArgs
from .elbo import elbo
from .utils import update_running
from .vae import VAE


def train(data: DataLoader, vae: VAE, optimizer: Optimizer, args: TrainArgs):
    """Bare bones VAE training loop"""
    step = 0
    smooth_loss = None
    vae.train()
    for _ in range(args.num_epochs):
        for x in data:
            kl_scale = args.kl_annealing.get_kl_scale()
            x = x[0]
            optimizer.zero_grad()
            loss, debug_info = elbo(x, vae(x), scale=kl_scale)
            loss.backward()
            optimizer.step()
            step += 1

            smooth_loss = update_running(smooth_loss, loss.item(), alpha=args.smoothing)
            if args.print_every and step % args.print_every == 0:
                print(
                    f"Step: {step} | ELBO: {smooth_loss:.5f} | KL scale: {kl_scale:.3f}"
                )
                if debug_info is not None:
                    print(debug_info)

        args.kl_annealing.step()
    return vae
