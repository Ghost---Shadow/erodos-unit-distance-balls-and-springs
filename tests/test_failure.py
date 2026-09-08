"""A snapped spring must exert exactly zero force."""

import numpy as np
import pytest

from erdos_unit_distance.energy import EnergyParams, energy_and_grad
from erdos_unit_distance.springs import LAWS, apply_failure, spring_energy_force

INF = float("inf")


def force_at(d, law="gaussian", lo=0.5, hi=2.0, depth=1.0, sigma=0.2):
    d = np.atleast_1d(np.asarray(d, dtype=float))
    e, f = spring_energy_force(d - 1.0, law, depth, sigma)
    return apply_failure(d - 1.0, e, f, law, depth, sigma, lo, hi)[1]


@pytest.mark.parametrize("law", LAWS)
def test_no_force_outside_the_failure_range(law):
    assert np.all(force_at([0.01, 0.2, 0.49], law=law) == 0.0)
    assert np.all(force_at([2.01, 3.0, 50.0], law=law) == 0.0)


def test_force_survives_inside_the_range():
    assert np.all(force_at([0.5, 0.7, 1.3, 2.0]) != 0.0)


@pytest.mark.parametrize("law", LAWS)
def test_energy_is_continuous_across_the_break(law):
    """No step in the energy means no spurious well at the threshold."""
    for edge in (0.5, 2.0):
        h = 1e-7
        inside = force_at([edge], law=law)          # touches the same code path
        assert inside is not None
        d = np.array([edge - h, edge + h])
        e, f = spring_energy_force(d - 1.0, law, 1.0, 0.2)
        e, _ = apply_failure(d - 1.0, e, f, law, 1.0, 0.2, 0.5, 2.0)
        assert e[0] == pytest.approx(e[1], abs=1e-5)


def test_failure_can_be_disabled():
    assert np.all(force_at([0.2, 5.0], law="hooke", lo=0.0, hi=INF) != 0.0)


def test_gradient_still_matches_with_failure_enabled():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(9, 2)) * 0.9
    p = EnergyParams(law="hooke", sigma=0.2, break_lo=0.5, break_hi=2.0)
    _, g = energy_and_grad(X, p)
    num = np.zeros_like(X)
    h = 1e-7
    for i in range(X.shape[0]):
        for k in range(2):
            up, dn = X.copy(), X.copy()
            up[i, k] += h
            dn[i, k] -= h
            num[i, k] = (energy_and_grad(up, p)[0] - energy_and_grad(dn, p)[0]) / (2 * h)
    assert np.abs(num - g).max() < 1e-5


def test_core_repulsion_is_not_subject_to_failure():
    """Balls closer than break_lo have snapped every spring between them;
    only the core is left to stop them merging."""
    X = np.array([[0.0, 0.0], [0.05, 0.0]])
    p = EnergyParams(r_min=0.3, break_lo=0.5, break_hi=2.0)
    _, g = energy_and_grad(X, p)
    assert g[0, 0] > 0.0      # pushed in -x, i.e. away from its neighbour
    assert g[1, 0] < 0.0


def test_failure_is_off_by_default():
    """The saturating laws already fade a violated spring out smoothly;
    the hard cutoff is a second, blunter mechanism, opt-in."""
    p = EnergyParams()
    assert p.break_lo == 0.0
    assert p.break_hi == INF

    X = np.array([[0.0, 0.0], [9.0, 0.0]])       # far past any sane break_hi
    _, g = energy_and_grad(X, EnergyParams(law="hooke", k_conf=0.0))
    assert np.abs(g).max() > 0.0                  # still pulling


def test_turning_failure_on_silences_a_far_pair():
    """Confinement is switched off here so the only force left could be
    the spring, and a snapped spring exerts none."""
    X = np.array([[0.0, 0.0], [9.0, 0.0]])
    _, g = energy_and_grad(X, EnergyParams(law="hooke", break_hi=2.0, k_conf=0.0))
    assert np.abs(g).max() == 0.0
