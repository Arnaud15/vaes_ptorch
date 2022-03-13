import hypothesis as hp
import hypothesis.strategies as st
import vaes_ptorch.annealing as anl


def test_lin_annealing(end_scale, zero_steps, lin_steps):
    annealing = anl.LinearAnnealing(
        end_scale=end_scale, zero_steps=zero_steps, linear_steps=lin_steps
    )
