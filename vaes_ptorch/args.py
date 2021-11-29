from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class DivAnnealing:
    """It is useful, to stabilize VAE training, to tune the scale of the
    divergence term in the VAE loss over time.


    This class supports the bare-bones scheduling approach
    - constant `start_scale` for `start_epochs`
    - linear progress from `start_scale` to `end_scale` for `linear_epochs`
    - constant `end_scale` thereafter
    """

    start_scale: float
    end_scale: float
    start_epochs: int
    linear_epochs: int

    def __post_init__(self):
        assert 0 <= self.start_epochs, self
        assert 0 <= self.linear_epochs, self
        assert self.start_scale >= 0.0, self
        assert self.end_scale >= 0.0, self
        self.epoch = 0

    def get_div_scale(self):
        if self.epoch >= (self.linear_epochs + self.start_epochs):
            return self.end_scale
        elif self.epoch >= self.start_epochs:
            if not self.linear_epochs:
                return self.end_scale
            else:
                progress = 1 + self.epoch - self.start_epochs
                todo = self.linear_epochs
                return (progress / todo) * (self.end_scale - self.start_scale) + self.start_scale
        else:
            return self.start_scale

    def step(self):
        self.epoch += 1


@dataclass(frozen=True)
class TrainArgs:
    num_epochs: int
    info_vae: bool = False
    print_every: int = 0  # never print if zero
    call_every: int = 0  # never use the callback if zero
    callback: Optional[Callable] = None
    smoothing: float = 0.9
    div_annealing: Optional[DivAnnealing] = None

    def __post_init__(self):
        if self.call_every:
            assert self.callback is not None
        if self.div_annealing is None:
            self.div_annealing = DivAnnealing(
                start_scale=1.0, end_scale=1.0, start_epochs=0, linear_epochs=0
            )
