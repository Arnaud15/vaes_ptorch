"""Utilities"""
import math
from typing import List, Optional

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import torch
import torchvision.transforms.functional as F  # type: ignore
from torch import Tensor
from torchvision.utils import make_grid  # type: ignore


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
