import dataclasses
import itertools
from typing import Callable, Dict, Iterator, Optional, Tuple

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader


@dataclasses.dataclass
class TrainArgs:
    """
    Parameters for a training job
    """

    num_steps: int
    print_every: int = 0  # never print if zero
    smoothing: float = 0.9

    def __post_init__(self):
        assert self.num_steps >= 0 and isinstance(self.num_steps, int), self
        assert self.print_every >= 0 and isinstance(self.print_every, int), self
        assert (
            self.smoothing > 0
            and self.smoothing < 1.0
            and isinstance(self.smoothing, float)
        ), self


def update_running(curr: Optional[float], obs: float, alpha: float = 0.9) -> float:
    """Update an exponentially weighted moving average with a new observation.

    If the current value of the moving average has not been initialized already
    it is `None` and set equal to the new observation."""

    assert alpha >= 0.0 and alpha < 1.0

    if curr is None:
        return obs
    else:
        return obs * (1.0 - alpha) + curr * alpha


def update_info_dict(info_dict: Dict[str, float], obs: Dict[str, float]):
    """Update the float values of a dictionary using an exponential moving average."""
    for key, obs_value in obs.items():
        updated_val = update_running(info_dict.get(key, None), obs_value)
        info_dict[key] = updated_val


def print_info_dict(info_dict: Dict[str, float]) -> str:
    """Formats and prints the float values held in a dictionary."""
    return " | ".join([f"{key}: {value:.5f}" for (key, value) in info_dict.items()])


def train_loop(
    net: torch.nn.Module,
    optimizer: Optimizer,
    loss_fn: Callable[..., Tuple[Tensor, Dict[str, float]]],
    args: TrainArgs,
    train_data: DataLoader,
) -> Dict[str, float]:
    """Training loop with periodic logging."""
    step = 1
    train_info = {}
    for batch in itertools.cycle(train_data):
        (loss, loss_info) = loss_fn(*net(batch))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        update_info_dict(train_info, loss_info)
        if args.print_every and step % args.print_every == 0:
            print(
                f"Training step: {step}/{args.num_steps} | {print_info_dict(train_info)}"
            )

        if step > args.num_steps:
            return train_info

        step += 1
    return train_info
