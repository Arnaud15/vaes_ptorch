"""MNIST experiments"""
import argparse
import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.utils.data as tdata
import torchvision

import vaes_ptorch.args as vae_args
import vaes_ptorch.models as models
import vaes_ptorch.train_vae as train_vae
import vaes_ptorch.utils as ut
import vaes_ptorch.vae as vae_nn

DATA_PATH = os.path.join(os.path.expanduser("~"), os.path.join("vaes_ptorch", "data"))

EXP_PATH = os.path.join(DATA_PATH, "experiments")


def mnist_main(args: argparse.Namespace):
    """Main script for MNIST experiments"""
    args, num_experiments = init_args(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for _ in range(args.num_repeats):
        for ix in range(num_experiments):
            mnist_experiment(
                exp_path=args.exp_path,
                device=device,
                info_vae=args.info_vae[ix],
                div_scale=args.div_scales[ix],
                latent_dim=args.latent_dims[ix],
                lr=args.lrs[ix],
                truncated_share=args.trunc_share,
                batch_size=args.batch_size,
                num_epochs=args.num_epochs,
            )


def init_args(args: argparse.Namespace):
    """Initialize experiment arguments and return (initialized arguments,
    number of experiments to run).

    1. If any of the arguments list is None, revert to the default value for
    this argument.
    2. Form all combination of argument lists to experiment over"""
    assert all(x in {0, 1} for x in args.info_vae), args.info_vae

    for att in ["div_scales", "latent_dims", "lrs"]:
        vals = getattr(args, att)
        if vals is None:
            setattr(args, att, [getattr(args, att[:-1])])
        else:
            assert len(vals) == len(
                set(vals)
            )  # values to experiment with must be unique

    num_experiments = (
        len(args.div_scales)
        * len(args.latent_dims)
        * len(args.lrs)
        * len(args.info_vae)
    )
    print(f"{num_experiments} experiments to run on MNIST")

    for att in ["div_scales", "latent_dims", "lrs", "info_vae"]:
        setattr(
            args,
            att,
            ut.repeat_list(
                getattr(args, att), num_experiments // len(getattr(args, att))
            ),
        )
        len(getattr(args, att)) == num_experiments

    print(f"experiments starting with args: {args}")
    return args, num_experiments


def mnist_experiment(
    exp_path: str,
    device: str,
    info_vae: bool,
    div_scale: float,
    latent_dim: int,
    lr: float,
    batch_size: int,
    num_epochs: int,
    eval_share: float = 0.3,
    truncated_share: float = 0.0,
):
    """Run a single MNIST experiment

    1. initialize training arguments
    2. get the MNIST train, validation and test data
    3. initialize the VAE model and optimizer
    4. train the VAE and measure its reconstruction loss on the validation set
    5. test the VAE and measure its reconstruction loss on the test set 
    6. save results and experiment arguments in a file
    """
    train_args = build_mnist_args(
        info_vae=info_vae, div_scale=div_scale, num_epochs=num_epochs
    )
    train_d, eval_d, test_d = build_mnist_data(
        batch_size=batch_size, eval_share=eval_share, truncated_share=truncated_share,
    )
    vae_net = build_mnist_vae(latent_dim=latent_dim, device=device)
    opt = torch.optim.Adam(params=vae_net.parameters(), lr=lr)
    _, eval_info = train_vae.train(
        train_data=train_d,
        vae=vae_net,
        optimizer=opt,
        train_args=train_args,
        eval_data=eval_d,
        device=device,
    )
    test_err = train_vae.evaluate(test_d, vae_net, device=device)
    ut.save_experiment(
        {
            "info_vae": info_vae,
            "div_scale": div_scale,
            "latent_dim": latent_dim,
            "lr": lr,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "eval_share": eval_share,
            "eval_error": eval_info["nll"],
            "test_error": test_err["nll"],
        },
        exp_path=exp_path,
    )


def build_mnist_vae(
    latent_dim: int,
    device: torch.device,
    hidden_size: int = 512,
    num_hidden_layers: int = 3,
) -> vae_nn.GaussianVAE:
    """Build a gaussian VAE where the encoder and decoder are simple MLPs."""
    encoder = nn.Sequential(
        nn.Flatten(),
        models.get_mlp(
            in_dim=28 * 28,
            out_dim=2 * latent_dim,
            h_dim=hidden_size,
            n_hidden=num_hidden_layers,
        ),
    )
    decoder = nn.Sequential(
        models.get_mlp(
            in_dim=latent_dim,
            out_dim=28 * 28,
            h_dim=hidden_size,
            n_hidden=num_hidden_layers,
        ),
        nn.Unflatten(1, (1, 28, 28)),
    )
    vae_net = vae_nn.GaussianVAE(
        encoder=encoder,
        decoder=decoder,
        latent_dim=latent_dim,
        obs_model=vae_nn.ObsModel.Bernoulli,
        device=torch.device(device),
        min_posterior_std=vae_args.MIN_STD,
    )
    print("vae model initialized")
    return vae_net


def build_mnist_args(
    info_vae: bool, num_epochs: int, div_scale: float
) -> vae_args.TrainArgs:
    """Initializes training arguments for the VAE experiment"""
    train_args = vae_args.TrainArgs(
        info_vae=info_vae,
        num_epochs=num_epochs,
        eval_every=1,
        target_div_scale=div_scale,
    )
    print(f"initialized training arguments to {train_args}")
    return train_args


def build_mnist_data(
    batch_size: int, eval_share: float, truncated_share: float = 0.0,
) -> Tuple[tdata.DataLoader, tdata.DataLoader, tdata.DataLoader]:
    """Extract a training, validation and test set for MNIST

    - truncated share must be a float in [0.0, 1.0) that controls what share of
      the mnist training and test data should be truncated in experiments
      (useful for testing, default to 0.0).

    Note: the MNIST digits are binarized so that our VAE decoder on MNIST will
    parameterize independent bernoulli random variables."""
    assert truncated_share >= 0.0 and truncated_share < 1.0
    train_set = torchvision.datasets.MNIST(
        root=DATA_PATH, train=True, download=True, transform=ut.binarize,
    )
    n = int(len(train_set) * (1.0 - truncated_share))
    train_set = tdata.Subset(train_set, list(range(n)))
    n_eval = int(n * eval_share)
    train_data, eval_data = tdata.random_split(
        train_set, [n - n_eval, n_eval], generator=torch.Generator().manual_seed(15),
    )
    train_loader = tdata.DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=True
    )
    eval_loader = tdata.DataLoader(
        dataset=eval_data, batch_size=batch_size, shuffle=True
    )
    test_set = torchvision.datasets.MNIST(
        root=DATA_PATH, train=False, download=True, transform=ut.binarize,
    )
    test_set = tdata.Subset(
        test_set, list(range(int(len(test_set) * (1.0 - truncated_share))))
    )
    test_loader = tdata.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=True
    )
    print(
        f"MNIST data extracted with {len(train_data)} training examples, {len(eval_data)} validation examples and {len(test_set)} test examples"
    )
    return train_loader, eval_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--info_vae",
        type=int,
        required=True,
        help="1 for InfoVAE, 0 for [Vanilla|Beta]VAE",
        nargs="+",
    )
    parser.add_argument(
        "--div_scales",
        type=float,
        required=False,
        help="Scaling factor for the KL or MMD divergence in the VAE loss",
        nargs="+",
    )
    parser.add_argument(
        "--div_scale",
        type=float,
        required=False,
        default=1.0,
        help="Scaling factor for the KL or MMD divergence in the VAE loss",
    )
    parser.add_argument(
        "--latent_dims",
        type=int,
        required=False,
        help="Size of the VAE's latent space",
        nargs="+",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        required=False,
        default=10,
        help="Size of the VAE's latent space",
    )
    parser.add_argument(
        "--lrs", type=float, required=False, help="Learning rates", nargs="+",
    )
    parser.add_argument(
        "--lr", type=float, required=False, default=1e-3, help="Learning rate",
    )
    parser.add_argument(
        "--num_repeats",
        type=int,
        default=1,
        help="Number of repetitions for each experiment.",
    )
    parser.add_argument(
        "--exp_path",
        type=str,
        default=EXP_PATH,
        help="Path to the directory where experiment results will be saved.",
    )
    parser.add_argument(
        "--trunc_share",
        type=float,
        default=0.0,
        help="Optional share (in [0.0, 1.0)) of the MNIST dataset to be truncated away for faster experiments",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Optional batch size parameter, defaults to 128",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=2,
        help="Optional num epochs parameter, defaults to 2",
    )
    mnist_main(parser.parse_args())
