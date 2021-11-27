from collections import namedtuple
from typing import Optional

import numpy as np
import scipy.stats as sts

Theta = namedtuple("Theta", ["mu", "sigma", "gamma"])


def expect(X: np.ndarray, theta: Theta) -> np.ndarray:
    n_mixtures = len(theta.gamma)
    x_given_theta_z = np.asarray(
        [
            sts.multivariate_normal.pdf(x=X, mean=theta.mu[j], cov=theta.sigma[j])
            for j in range(n_mixtures)
        ]
    ).T  # shape (n,m)
    assert x_given_theta_z.shape == (X.shape[0], n_mixtures)
    probas_x_z_given_theta = theta.gamma * x_given_theta_z
    return probas_x_z_given_theta / probas_x_z_given_theta.sum(
        1, keepdims=True
    )  # shape (n,m)


def maximize(X: np.ndarray, soft_zs: np.ndarray, theta: Theta) -> Theta:
    n_mixtures = len(theta.gamma)
    assert soft_zs.shape[1] == n_mixtures
    assert len(soft_zs) == len(X)
    new_mus = []
    new_sigmas = []
    for j in range(n_mixtures):
        probas_j = soft_zs[:, j]
        sum_j = probas_j.sum()
        mu_j = np.einsum("i,id->d", probas_j, X) / sum_j
        sig_j = np.einsum("a,ab,ac->bc", probas_j, X - mu_j, X - mu_j) / sum_j
        new_mus.append(mu_j)
        new_sigmas.append(sig_j)
    return Theta(mu=new_mus, sigma=new_sigmas, gamma=soft_zs.mean(0))


def init_theta(
    n_mixtures: int,
    dim: int,
    means: Optional[np.ndarray] = None,
    scale: Optional[np.ndarray] = None,
) -> Theta:
    if means is not None:
        assert means.shape == (dim,)
    else:
        means = np.zeros(dim)
    if scale is not None:
        assert scale.shape == (dim,)
        scale = np.diagflat(scale)
    else:
        scale = np.eye(dim)
    return Theta(
        mu=[np.random.randn(dim) + means for _ in range(n_mixtures)],
        sigma=[scale for _ in range(n_mixtures)],
        gamma=np.ones(n_mixtures) / n_mixtures,
    )


def em_gaussian_mixture(X: np.ndarray, n_mixtures: int, num_steps: int = 100) -> Theta:
    n, dim = X.shape
    theta = init_theta(n_mixtures, dim, X.mean(0), X.std(0))
    for _ in range(num_steps):
        soft_zs = expect(X, theta)
        theta = maximize(X, soft_zs, theta)
    return theta
