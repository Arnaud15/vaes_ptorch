import math

import hypothesis as hp
import hypothesis.strategies as st
import torch

import vaes_ptorch.args as args
import vaes_ptorch.proba as proba

dim_strategy = st.integers(min_value=2, max_value=100)

pos_finite_floats = st.floats(min_value=1e-3, allow_nan=False, allow_infinity=False)
finite_floats = st.floats(allow_nan=False, allow_infinity=False)


@hp.given(dim_strategy, dim_strategy, dim_strategy, pos_finite_floats)
def test_rbf_kernel(n_left, n_right, dim, bandwidth):
    p_samples = proba.sample_gaussian(
        proba.NormalParams(mu=torch.zeros(dim), var=torch.ones(dim)), n_samples=n_left
    )
    q_samples = proba.sample_gaussian(
        proba.NormalParams(mu=torch.ones(dim), var=torch.ones(dim)), n_samples=n_right
    )
    kernel_pp = proba.rbf_kernel(p_samples, p_samples, bandwidth)
    kernel_pq = proba.rbf_kernel(p_samples, q_samples, bandwidth)
    kernel_qp = proba.rbf_kernel(q_samples, p_samples, bandwidth)
    kernel_pp_higher_band = proba.rbf_kernel(p_samples, p_samples, 2 * bandwidth)
    assert kernel_pp <= kernel_pp_higher_band
    assert abs(kernel_pq - kernel_qp) <= 1e-5
    assert kernel_pq <= kernel_pp


@hp.given(dim_strategy)
@hp.settings(max_examples=10)
def test_mmd_rbf_deterministic_target(dim):
    n_samples = 128
    epsilon = 0.1
    target = torch.ones(n_samples, dim)
    candidates = [
        proba.sample_gaussian(
            proba.NormalParams(
                mu=torch.ones(dim), var=torch.ones(dim) * (i + 1) * epsilon
            ),
            n_samples=n_samples,
        )
        for i in range(10)
    ]
    divergences = [proba.mmd_rbf(target, candidate) for candidate in candidates]
    assert min(divergences) == divergences[0]


large_dim_strategy = st.integers(min_value=50, max_value=100)


@hp.given(large_dim_strategy)
@hp.settings(max_examples=10)
def test_mmd_rbf_gaussian_target(dim):
    n_samples = 128
    target = proba.sample_gaussian(
        proba.NormalParams(mu=torch.zeros(dim), var=torch.ones(dim)),
        n_samples=n_samples,
    )

    epsilon = 0.1
    candidates = [
        proba.sample_gaussian(
            proba.NormalParams(
                mu=torch.zeros(dim), var=torch.ones(dim) * (i + 1) * epsilon
            ),
            n_samples=n_samples,
        )
        for i in range(5)
    ]
    divergences = [proba.mmd_rbf(target, candidate) for candidate in candidates]
    assert min(divergences) == divergences[-1]


@hp.given(dim_strategy, dim_strategy)
def test_kl_div(n_points, dim):
    mu1 = torch.randn(n_points, dim)
    mu2 = torch.rand(n_points, dim) + mu1
    mu3 = torch.rand(n_points, dim) + mu2
    var = torch.rand(n_points, dim) + args.MIN_VAR
    kl_11 = proba.gaussian_kl(var, mu1, var, mu1)
    kl_12 = proba.gaussian_kl(var, mu1, var, mu2)
    kl_13 = proba.gaussian_kl(var, mu1, var, mu3)
    assert kl_11 == 0.0
    assert kl_11 >= 0
    assert kl_12 >= 0
    assert kl_12 <= kl_13
