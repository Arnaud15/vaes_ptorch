from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .args import TrainArgs
from .losses import elbo_loss, info_vae_loss
from .utils import update_running
from .vae import GaussianVAE


def train(data: DataLoader, vae: GaussianVAE, optimizer: Optimizer, args: TrainArgs):
    """Bare bones VAE training loop"""
    step = 0
    smooth_loss = None
    vae.train()
    for _ in range(args.num_epochs):
        for x in data:
            div_scale = args.div_annealing.get_div_scale()  # type: ignore
            x = x[0]
            # print(x.shape)
            optimizer.zero_grad()
            if args.info_vae:
                loss, debug_info = info_vae_loss(x, vae(x), scale=div_scale)
            else:
                loss, debug_info = elbo_loss(x, vae(x), scale=div_scale)
            loss.backward()
            optimizer.step()

            smooth_loss = update_running(smooth_loss, loss.item(), alpha=args.smoothing)
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
