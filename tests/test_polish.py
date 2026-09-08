"""The exact projection must be exact, and must keep the balls distinct."""

import numpy as np
import pytest

from erdos_unit_distance import init as I
from erdos_unit_distance.counting import count_unit_distances, min_separation, unit_edges
from erdos_unit_distance.polish import polish, polish_and_prune, residuals


def test_perturbed_lattice_snaps_back_to_machine_precision():
    rng = np.random.default_rng(3)
    X = I.triangular_lattice(12) + rng.normal(scale=0.03, size=(12, 2))
    edges = unit_edges(X, 0.15)
    Xp, kept, err = polish_and_prune(X, edges)
    assert kept.shape[0] == edges.shape[0]
    assert err < 1e-12
    assert min_separation(Xp) == pytest.approx(1.0, abs=1e-9)


def test_polish_refuses_to_merge_points():
    """Three balls asked to be mutually unit-distant from a fourth pair can
    only satisfy every equation by collapsing; the core must prevent it."""
    X = np.array([[0.0, 0.0], [0.02, 0.0], [0.01, 0.9], [0.01, -0.9]])
    edges = np.array([[0, 2], [1, 2], [0, 3], [1, 3]])
    Xp, _ = polish(X, edges, r_min=0.3)
    assert min_separation(Xp) > 0.25


def test_disabling_the_core_does_allow_the_degenerate_collapse():
    """The counting artefact the barrier exists to prevent, made explicit.

    Balls 0 and 1 are both required to sit a unit distance from three
    others.  The set of points a unit distance from two fixed balls has
    only two members, so three such constraints can only be met by making
    some pair coincide -- which is exactly the fake configuration that
    scores ~n^2/3 spurious "unit distances".
    """
    X = np.array([[0.01, 0.0], [-0.01, 0.005],
                  [1.0, 0.0], [0.0, 1.0], [-0.7, -0.7]])
    edges = np.array([[0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4]])

    loose, err_loose = polish(X, edges, r_min=0.0)
    assert err_loose < 1e-9                    # every equation satisfied
    assert min_separation(loose) < 1e-3        # ...only by merging a pair

    tight, _ = polish(X, edges, r_min=0.3)
    assert min_separation(tight) > 0.25        # the barrier refuses


def test_empty_edge_set_is_a_no_op():
    X = I.random_disc(5, np.random.default_rng(0))
    Xp, err = polish(X, np.zeros((0, 2), dtype=np.int64))
    assert err == 0.0
    assert np.allclose(Xp, X)


def test_unrealisable_edges_get_pruned():
    """Four balls cannot be pairwise unit-distant in the plane (that needs
    a tetrahedron), so at least one edge has to be dropped."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, 2)) * 0.5
    edges = np.array([[i, j] for i in range(4) for j in range(i + 1, 4)])
    _, kept, err = polish_and_prune(X, edges, r_min=0.3)
    assert kept.shape[0] < edges.shape[0]
    assert err < 1e-9


def test_matrix_free_path_agrees_with_the_dense_one():
    from erdos_unit_distance import polish as polish_mod

    rng = np.random.default_rng(5)
    X = I.triangular_lattice(40) + rng.normal(scale=0.02, size=(40, 2))
    edges = unit_edges(X, 0.1)

    original = polish_mod.DENSE_LIMIT
    try:
        polish_mod.DENSE_LIMIT = 10_000
        dense, e_dense = polish(X, edges)
        polish_mod.DENSE_LIMIT = 1          # force the CG path
        sparse, e_sparse = polish(X, edges)
    finally:
        polish_mod.DENSE_LIMIT = original

    assert e_dense < 1e-10 and e_sparse < 1e-10
    assert count_unit_distances(dense, 1e-9) == count_unit_distances(sparse, 1e-9)
