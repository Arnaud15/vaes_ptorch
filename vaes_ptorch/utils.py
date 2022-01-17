"""Utilities"""
import collections
import json
import math
import os
import random
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F  # type: ignore
from torch import Tensor
from torchvision.utils import make_grid  # type: ignore


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


def bits_per_dim_multiplier(dims: List[int]) -> float:
    """Computes the product of input dimensions x `log(2)` to rescale log
    likelihood numbers to "bits per dimension" metrics."""
    assert all(x > 0 for x in dims)
    dims_prod = torch.prod(torch.tensor(dims)).item()
    return math.log(2.0) * dims_prod


def update_running(curr: Optional[float], obs: float, alpha: float) -> float:
    """Update an exponentially weighted moving average with a new observation.
    
    If the current value of the moving average has not been initialized already
    it is `None` and set equal to the new observation."""

    assert alpha >= 0.0 and alpha < 1.0

    if curr is None:
        return obs
    else:
        return obs * (1.0 - alpha) + curr * alpha


def show(img: Tensor):
    """Small utility to plot a tensor of images"""
    img = img.detach()
    try:
        img = F.to_pil_image(img)
    except ValueError:  # handle batched images to plot
        img = make_grid(img)
        img = F.to_pil_image(img)
    plt.imshow(np.asarray(img))
    plt.xticks([])  # remove pyplot borders
    plt.yticks([])
    plt.show()
    plt.close()
