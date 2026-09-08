"""Known configurations have known unit-distance counts."""

import numpy as np
import pytest

from erdos_unit_distance import init as I
from erdos_unit_distance.counting import (count_unit_distances, degree_sequence,
                                          min_separation, unit_edges)


def test_moser_spindle_has_eleven_unit_distances():
    assert count_unit_distances(I.moser_spindle(), 1e-9) == 11


def test_moser_spindle_points_are_distinct():
    assert min_separation(I.moser_spindle()) > 0.4


def test_hexagonal_flower_attains_the_known_optimum_for_seven_points():
    assert count_unit_distances(I.triangular_lattice(7), 1e-9) == 12


def test_triangular_lattice_approaches_three_n():
    """Every interior ball has six unit neighbours, so the count is 3n - O(sqrt n)."""
    for n in (50, 100, 200):
        c = count_unit_distances(I.triangular_lattice(n), 1e-9)
        assert 3 * n - 4 * np.sqrt(n) < c < 3 * n


def test_erdos_grid_overtakes_the_triangular_lattice():
    """The whole point of the sqrt(n) x sqrt(n) construction."""
    assert count_unit_distances(I.erdos_grid(100), 1e-9) > \
           count_unit_distances(I.triangular_lattice(100), 1e-9)


def test_edges_are_ordered_pairs_without_duplicates():
    X = I.triangular_lattice(12)
    e = unit_edges(X, 1e-9)
    assert (e[:, 0] < e[:, 1]).all()
    assert len({tuple(p) for p in e.tolist()}) == e.shape[0]


def test_degree_sequence_sums_to_twice_the_edge_count():
    X = I.triangular_lattice(19)
    e = unit_edges(X, 1e-9)
    assert degree_sequence(19, e).sum() == 2 * e.shape[0]


def test_a_wider_tolerance_never_finds_fewer_edges():
    rng = np.random.default_rng(0)
    X = I.random_disc(30, rng)
    counts = [count_unit_distances(X, t) for t in (1e-6, 1e-3, 1e-2, 1e-1)]
    assert counts == sorted(counts)
