from dataclasses import dataclass
from typing import Optional


@dataclass
class KLAnnealing:
    num_epochs: int
    reconstruction_epochs: int
    kl_warmup_epochs: int

    def __post_init__(self):
        assert 0 <= self.reconstruction_epochs, self
        assert 0 <= self.kl_warmup_epochs, self
        self.epoch = 0

    def get_kl_scale(self):
        if self.epoch >= (self.kl_warmup_epochs + self.reconstruction_epochs):
            return 1.0
        elif self.epoch >= self.reconstruction_epochs:
            if not self.kl_warmup_epochs:
                return 1.0
            else:
                progress = 1 + self.epoch - self.reconstruction_epochs
                todo = self.kl_warmup_epochs
                return progress / todo
        else:
            return 0.0

    def step(self):
        self.epoch += 1


@dataclass
class TrainArgs:
    num_epochs: int
    print_every: int = 0  # never print if zero
    smoothing: float = 0.9
    kl_annealing: Optional[KLAnnealing] = None

    def __post_init__(self):
        if self.kl_annealing is None:
            self.kl_annealing = KLAnnealing(
                num_epochs=self.num_epochs, reconstruction_epochs=0, kl_warmup_epochs=0
            )
        else:
            assert self.kl_annealing.num_epochs == self.num_epochs
