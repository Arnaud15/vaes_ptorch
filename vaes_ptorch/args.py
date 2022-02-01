from dataclasses import dataclass
from typing import Optional

import vaes_ptorch.annealing as annealing

# Minimum variance for multivariate gaussian models, to avoid pathological
# overfitting situations.
MIN_VAR = 1e-10


@dataclass
class TrainArgs:
    """
    VAE training parameters for an experiment
    """

    num_epochs: int
    info_vae: bool = False
    print_every: int = 0  # never print if zero
    eval_every: int = 0  # never eval if zero
    smoothing: float = 0.9
    target_div_scale: float = 1.0
    soft_free_bits: bool = False
    zero_div_steps: int = 0
    lin_annealing_steps: int = 0
    div_annealing: Optional[annealing.AnnealingSchedule] = None

    def __post_init__(self):
        assert self.num_epochs >= 0, self
        assert self.print_every >= 0, self
        assert self.eval_every >= 0, self
        assert self.smoothing > 0 and self.smoothing < 1.0, self
        assert self.target_div_scale >= 0.0, self
        assert self.zero_div_steps >= 0.0, self
        assert self.lin_annealing_steps >= 0.0, self
        if self.soft_free_bits:
            self.div_annealing = annealing.SoftFreeBits(
                target_lambda=self.target_div_scale
            )
        else:
            self.div_annealing = annealing.LinearAnnealing(
                end_scale=self.target_div_scale,
                zero_steps=self.zero_div_steps,
                linear_steps=self.lin_annealing_steps,
            )
