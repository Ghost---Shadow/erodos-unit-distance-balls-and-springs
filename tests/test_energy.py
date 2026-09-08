"""The gradient really is the gradient, for every spring law."""

import numpy as np
import pytest

from erdos_unit_distance.energy import EnergyParams, energy_and_grad, soft_count
from erdos_unit_distance.springs import LAWS, spring_energy_force


def numeric_grad(X, p, h=1e-6):
    g = np.zeros_like(X)
    for i in range(X.shape[0]):
        for k in range(2):
            up, dn = X.copy(), X.copy()
            up[i, k] += h
            dn[i, k] -= h
            g[i, k] = (energy_and_grad(up, p)[0] - energy_and_grad(dn, p)[0]) / (2 * h)
    return g


@pytest.mark.parametrize("law", LAWS)
def test_gradient_matches_finite_differences(law):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(9, 2)) * 0.8
    p = EnergyParams(law=law, sigma=0.2)
    _, g = energy_and_grad(X, p)
    assert np.abs(numeric_grad(X, p) - g).max() < 1e-6


@pytest.mark.parametrize("law", LAWS)
def test_gradient_with_overlapping_balls(law):
    """The core repulsion is active here, so it gets differentiated too."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(7, 2)) * 0.05
    p = EnergyParams(law=law, sigma=0.15, r_min=0.2)
    _, g = energy_and_grad(X, p)
    assert np.abs(numeric_grad(X, p) - g).max() < 1e-5


def test_row_blocking_does_not_change_the_answer():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(37, 2))
    e1, g1 = energy_and_grad(X, EnergyParams(block=4))
    e2, g2 = energy_and_grad(X, EnergyParams(block=1024))
    assert e1 == pytest.approx(e2, rel=1e-12)
    assert np.abs(g1 - g2).max() < 1e-12


@pytest.mark.parametrize("law", ["gaussian", "lorentz"])
def test_saturating_laws_have_sigma_independent_depth(law):
    """The well depth must not shrink with sigma, or the objective
    evaporates exactly when the anneal needs it most."""
    s = np.array([50.0])
    depths = [spring_energy_force(s, law, 1.0, sig)[0][0] for sig in (0.4, 0.05, 0.005)]
    assert all(d == pytest.approx(1.0, abs=1e-3) for d in depths)


@pytest.mark.parametrize("law", LAWS)
def test_satisfied_spring_costs_nothing(law):
    e, f = spring_energy_force(np.zeros(1), law, 1.0, 0.1)
    assert e[0] == pytest.approx(0.0)
    assert f[0] == pytest.approx(0.0)


@pytest.mark.parametrize("law", LAWS)
def test_compressed_spring_pushes_apart_and_stretched_pulls_together(law):
    _, f = spring_energy_force(np.array([-0.05, 0.05]), law, 1.0, 0.2)
    assert f[0] < 0.0     # dE/dd < 0, so the force increases the distance
    assert f[1] > 0.0


def test_soft_count_approaches_the_hard_count():
    from erdos_unit_distance.counting import count_unit_distances
    from erdos_unit_distance.init import triangular_lattice

    X = triangular_lattice(10)
    assert soft_count(X, 0.01) == pytest.approx(count_unit_distances(X, 1e-9), abs=0.5)
