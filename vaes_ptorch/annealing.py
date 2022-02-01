"""To stabilize the training of a VAE, it is useful to anneal the scale of the
prior regularization term in the ELBO.

This module implements functionality to anneal this term during VAE
training."""

from dataclasses import dataclass
from typing import Any, Protocol


class AnnealingSchedule(Protocol):
    def step(self):
        """Increment the annealing schedule by 1 step"""
        ...

    def get_scale(self, info: Any) -> float:
        """Return the current prior regularization scale"""
        ...


@dataclass
class LinearAnnealing:
    """
    This class supports a bare-bones scheduling approach for this divergence scale:
    - constant at 0.0 for `zero_steps`
    - linear progress from 0.0 to `end_scale` for `num_steps`
    - constant `end_scale` thereafter
    """

    end_scale: float = 1.0
    zero_steps: int = 0
    linear_steps: int = 0

    def __post_init__(self):
        assert 0 <= self.zero_steps, self
        assert 0 <= self.linear_steps, self
        assert self.end_scale >= 0.0, self
        self.curr_step = 0

    def get_scale(self, _info: Any) -> float:
        return min(
            self.end_scale,
            max(
                (self.curr_step - self.zero_steps)
                * (self.end_scale / self.linear_steps),
                0.0,
            ),
        )

    def step(self):
        self.curr_step += 1


class SoftFreeBits:
    """Described in the appendix of https://arxiv.org/abs/1611.02731"""

    target_lambda: float
    scale: float = 1e-5
    lambda_tol_pct: float = 30.0
    correction_pct: float = 10.0

    def __post_init__(self):
        assert 0 < self.scale, self
        assert 0 < self.lambda_tol_pct < 100, self
        assert 0 < self.correction_pct < 100, self

    def step(self, div_val: float):
        if div_val > self.target_lambda * (1.0 + self.lambda_tol_pct / 100.0):
            # too much info in the posterior, increase the penalty
            self.scale *= 1.0 + self.correction_pct / 100.0
        if div_val < self.target_lambda:
            # not enough info in the posterior, decrease the penalty
            self.scale *= 1.0 - self.correction_pct / 100.0
