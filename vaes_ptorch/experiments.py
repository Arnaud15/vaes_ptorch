"""MNIST experiments"""
import argparse
import collections
import json
import os
import random
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.utils.data as tdata
import torchvision
import torchvision.transforms as T

import vaes_ptorch.args as vae_args
import vaes_ptorch.losses as losses
import vaes_ptorch.models as models
import vaes_ptorch.train_vae as train_vae
import vaes_ptorch.vae as vaes

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
                info_vae=args.info_vae,
                div_scale=args.div_scales[ix],
                latent_dim=args.latent_dims[ix],
                lr=args.lrs[ix],
                truncated_share=args.trunc_share,
            )


def init_args(args: argparse.Namespace):
    """Initialize experiment arguments and return (initialized arguments,
    number of experiments to run).

    1. If any of the arguments list is None, revert to the default value for
    this argument.
    2. Form all combination of argument lists to experiment over"""
    assert args.info_vae in {0, 1}

    for att in ["div_scales", "latent_dims", "lrs"]:
        vals = getattr(args, att)
        if vals is None:
            setattr(args, att, [getattr(args, att[:-1])])
        else:
            assert len(vals) == len(
                set(vals)
            )  # values to experiment with must be unique

    num_experiments = len(args.div_scales) * len(args.latent_dims) * len(args.lrs)
    print(f"{num_experiments} experiments to run on MNIST")

    for att in ["div_scales", "latent_dims", "lrs"]:
        setattr(
            args,
            att,
            repeat_list(getattr(args, att), num_experiments // len(getattr(args, att))),
        )
        len(getattr(args, att)) == num_experiments

    print(f"experiments starting with args: {args}")
    return args, num_experiments


def repeat_list(input_list: List[Any], num_repeats: int) -> List[Any]:
    """Create and return a new list formed from the repetition of an input list
    for a specified number of times."""
    assert num_repeats > 0, num_repeats
    res = []
    for _ in range(num_repeats):
        res += input_list
    assert len(res) == len(input_list) * num_repeats, (
        len(res),
        len(input_list),
        num_repeats,
    )
    return res


def mnist_experiment(
    exp_path: str,
    device: str,
    info_vae: bool,
    div_scale: float,
    latent_dim: int,
    lr: float,
    batch_size: int = 128,
    num_epochs: int = 15,
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
    vae_nn = build_mnist_vae(latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(params=vae_nn.parameters(), lr=lr)
    eval_err = train_vae.train(
        train_data=train_d,
        vae=vae_nn,
        optimizer=opt,
        args=train_args,
        eval_data=eval_d,
        device=device,
    ).eval_ewma
    test_err = train_vae.evaluate(test_d, vae_nn, args=train_args, device=device)
    save_experiment(
        {
            "info_vae": info_vae,
            "div_scale": div_scale,
            "latent_dim": latent_dim,
            "lr": lr,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "eval_share": eval_share,
            "eval_error": eval_err,
            "test_error": test_err,
        },
        exp_path=exp_path,
    )


def save_experiment(exp_data: Dict[str, Any], exp_path: str):
    """Save experiments data in JSON format
    - The JSON file name is a random integer
    - The JSON file is saved in the folder specified by exp_path
    """
    check_exp_dir(exp_path)
    filename = random_filename(exp_path)
    filepath = os.path.join(exp_path, filename)
    with open(filepath, "w") as exp_file:
        json.dump(exp_data, exp_file)
        print(f"saved experiments data {exp_data} at {filepath}")


def random_filename(exp_path: str):
    """Small utility to generate a random filename before saving an experiment,
    dealing naively with collisions.

    Assumes that fewer than 1_000_000_000 files are already saved in
    `exp_path`."""
    occupied_filenames = set(
        name for name in os.listdir(exp_path) if name.endswith(".json")
    )
    filename = str(random.randint(0, 1_000_000_000)) + ".json"
    while filename in occupied_filenames:
        filename = str(random.randint(0, 1_000_000_000)) + ".json"
    return filename


def load_experiments_data(exp_path: str) -> Dict[str, List[Any]]:
    """Load experiments data from the JSON files stored in the experiments folder"""
    full_data = collections.defaultdict(list)
    exp_filenames = [
        string for string in os.listdir(exp_path) if string.endswith(".json")
    ]
    print(f"{len(exp_filenames)} experiment files to collate")
    for filename in exp_filenames:
        filepath = os.path.join(exp_path, filename)
        with open(filepath, "r") as exp_file:
            data = json.load(exp_file)
            for key, value in data.items():
                full_data[key].append(value)
    return full_data


def binarize(x):
    """Useful torchvision transformation which converts grayscale pixel values
    in [0, 1] to binary data in {0, 1}."""
    tensor = T.ToTensor()(x)
    mask = tensor > 0.5
    tensor[mask] = 1.0
    tensor[~mask] = 0.0
    return tensor


def check_exp_dir(exp_path: str):
    """Create a directory if it does not already exist."""
    if not os.path.isdir(exp_path):
        print("creating experiments directory")
        os.mkdir(exp_path)
    else:
        print("experiments directory already exists")


def build_mnist_vae(
    latent_dim: int, hidden_size: int = 512, num_hidden_layers: int = 3
) -> vaes.GaussianVAE:
    """Build a gaussian VAE where the encoder and decoder are simple MLPs."""
    encoder = vaes.GaussianModel(
        model=nn.Sequential(
            nn.Flatten(),
            models.get_mlp(
                in_dim=28 * 28,
                out_dim=2 * latent_dim,
                h_dims=[hidden_size] * num_hidden_layers,
            ),
        ),
        out_dim=latent_dim,
        min_var=1e-10,
    )
    decoder = vaes.GaussianModel(
        model=nn.Sequential(
            models.get_mlp(
                in_dim=latent_dim,
                out_dim=2 * 28 * 28,
                h_dims=[hidden_size] * num_hidden_layers,
            ),
            nn.Unflatten(1, (2, 28, 28)),
        ),
        out_dim=1,
        split_dim=1,
    )
    vae_nn = vaes.GaussianVAE(encoder=encoder, decoder=decoder)
    print("vae model initialized")
    return vae_nn


def build_mnist_args(
    info_vae: bool, num_epochs: int, div_scale: float
) -> vae_args.TrainArgs:
    """Initializes training arguments for the VAE experiment"""
    train_args = vae_args.TrainArgs(
        likelihood=losses.Likelihood.Bernoulli,
        info_vae=info_vae,
        num_epochs=num_epochs,
        div_annealing=vae_args.DivAnnealing(
            start_epochs=5, linear_epochs=5, start_scale=0.0, end_scale=div_scale,
        ),
        print_every=0,
        eval_every=1,
        smoothing=0.9,
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
        root=DATA_PATH, train=True, download=True, transform=binarize,
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
        root=DATA_PATH, train=False, download=True, transform=binarize,
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
    mnist_main(parser.parse_args())
