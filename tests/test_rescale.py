"""The Erdos rescaling trick, separated from his grid."""

import numpy as np
import pytest

from erdos_unit_distance import init as I
from erdos_unit_distance.counting import audit, count_unit_distances, min_separation

TOL = 1e-9


def test_rescaling_makes_the_commonest_distance_unit():
    P = I.triangular_lattice(300)
    Q = I.rescale_to_popular(P)
    mult, dist = I.distance_spectrum(P, top=1)[0]
    assert dist == pytest.approx(np.sqrt(7.0), abs=1e-6)   # a Loeschian norm
    assert count_unit_distances(Q, TOL) == mult


def test_rescaling_is_a_similarity_so_it_cannot_merge_points():
    """It only scales, so a valid configuration stays valid."""
    P = I.triangular_lattice(300)
    Q = I.rescale_to_popular(P)
    result = audit(Q, epsilon=TOL)
    assert result.ok
    assert result.distinct_points == 300
    assert min_separation(Q) > 0


def test_eisenstein_grid_beats_the_square_grid_version():
    """Loeschian norms i^2+ij+j^2 spike harder than i^2+j^2."""
    for n in (100, 300, 900):
        assert count_unit_distances(I.eisenstein_grid(n), TOL) > \
               count_unit_distances(I.erdos_grid(n), TOL)


def test_eisenstein_grid_beats_the_plain_triangular_lattice():
    for n in (300, 900):
        assert count_unit_distances(I.eisenstein_grid(n), TOL) > \
               count_unit_distances(I.triangular_lattice(n), TOL)


def test_rescaling_does_nothing_to_flower_sums():
    """Generic rotations shatter the distance spectrum, so the unit
    distance is already the tallest spike and there is nothing to gain."""
    P = I.hex_minkowski(343)
    before = count_unit_distances(P, TOL)
    after = count_unit_distances(I.rescale_to_popular(P), TOL)
    assert after == before
    assert I.distance_spectrum(P, top=1)[0][1] == pytest.approx(1.0, abs=1e-6)


def test_rescaling_gain_tracks_how_concentrated_the_spectrum_is():
    """The trick and the Minkowski-basis trick pull in opposite
    directions: one wants a concentrated spectrum, the other spreads it."""
    def distinct_distances(P):
        return len(I.distance_spectrum(P, top=10 ** 6))

    def gain(P):
        return count_unit_distances(I.rescale_to_popular(P), TOL) / \
            max(count_unit_distances(P, TOL), 1)

    concentrated = I.triangular_lattice(900)
    spread = I.hex_minkowski(900)
    assert distinct_distances(concentrated) < distinct_distances(spread)
    assert gain(concentrated) > gain(spread)


def test_spectrum_is_sorted_by_multiplicity():
    rows = I.distance_spectrum(I.triangular_lattice(300), top=5)
    assert [m for m, _ in rows] == sorted([m for m, _ in rows], reverse=True)


def test_available_as_an_init():
    X = I.make("eisenstein_grid", 60, np.random.default_rng(0))
    assert X.shape == (60, 2)
