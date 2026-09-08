"""The centered hexagon and the constructions built from it."""

import numpy as np
import pytest

from erdos_unit_distance import init as I
from erdos_unit_distance.counting import audit, count_unit_distances
from erdos_unit_distance.known import known_optimum


def test_centered_hexagon_is_the_optimum_for_seven_points():
    """Hub plus six unit vectors: 6 spokes + 6 rim = 12 = u(7)."""
    H = I.centered_hexagon()
    assert H.shape == (7, 2)
    assert count_unit_distances(H, 1e-9) == 12 == known_optimum(7)


def test_rotating_the_hexagon_does_not_change_its_count():
    for theta in (0.0, 0.3, 1.0):
        assert count_unit_distances(I.centered_hexagon(theta), 1e-9) == 12


@pytest.mark.parametrize("k", [1, 2, 3])
def test_minkowski_flowers_hit_the_closed_form(k):
    """k rotated flowers give 7^k points and exactly 12*k*7^(k-1) edges."""
    angles = (0.0, 0.3, 0.7)[:k]
    P = I.hex_minkowski(7 ** k, angles=angles)
    assert P.shape[0] == 7 ** k
    assert count_unit_distances(P, 1e-9) == 12 * k * 7 ** (k - 1)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_minkowski_flowers_have_no_overlapping_points(k):
    """The count is only meaningful if the summands stayed distinct."""
    angles = (0.0, 0.3, 0.7)[:k]
    result = audit(I.hex_minkowski(7 ** k, angles=angles))
    assert result.ok, result.problems
    assert result.distinct_points == 7 ** k


def test_unrotated_sums_collapse_back_to_the_triangular_lattice():
    """The six hexagon vectors generate the lattice, so rotating is the
    whole point of the construction."""
    P = I.hex_minkowski(19, angles=(0.0, 0.0))
    assert P.shape[0] == 19
    assert count_unit_distances(P, 1e-9) == \
           count_unit_distances(I.triangular_lattice(19), 1e-9)


def test_minkowski_beats_the_triangular_lattice_at_its_natural_size():
    for k in (2, 3):
        n = 7 ** k
        mk = count_unit_distances(I.hex_minkowski(n, angles=(0.0, 0.3, 0.7)[:k]), 1e-9)
        assert mk > count_unit_distances(I.triangular_lattice(n), 1e-9)


def test_hex_minkowski_is_available_as_an_init():
    rng = np.random.default_rng(0)
    X = I.make("hex_minkowski", 40, rng)
    assert X.shape == (40, 2)
