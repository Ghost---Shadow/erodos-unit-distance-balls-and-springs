"""Overlapping points must never inflate a reported count."""

import numpy as np
import pytest

from erdos_unit_distance import init as I
from erdos_unit_distance.counting import audit, merge_duplicates
from erdos_unit_distance.solver import solve_once


def clustered(cluster_sizes, spread=0.0, seed=0):
    """Coincident (or nearly coincident) clusters on a unit triangle."""
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    rng = np.random.default_rng(seed)
    pts = []
    for corner, size in zip(corners, cluster_sizes):
        pts.append(corner + rng.normal(scale=spread, size=(size, 2)))
    return np.vstack(pts)


def test_audit_rejects_exactly_coincident_clusters():
    """The cheat: 12 balls stacked on 3 spots score 48 -- well past u(12)=27."""
    X = clustered((4, 4, 4))
    result = audit(X)
    assert not result.ok
    assert result.edges_claimed == 48
    assert result.distinct_points == 3
    assert result.edges_among_distinct_points == 3
    assert any("coincide" in p for p in result.problems)


def test_audit_accepts_a_genuine_configuration():
    result = audit(I.triangular_lattice(19))
    assert result.ok
    assert result.problems == []
    assert result.distinct_points == 19
    assert result.edges_claimed == 42


def test_audit_catches_an_overstated_edge_list():
    X = I.triangular_lattice(7)
    liar = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
    result = audit(X, liar)
    assert not result.ok


def test_audit_catches_duplicate_entries_in_the_edge_list():
    X = I.triangular_lattice(7)
    from erdos_unit_distance.counting import unit_edges
    e = unit_edges(X, 1e-9)
    result = audit(X, np.vstack([e, e[:1]]))
    assert not result.ok
    assert any("duplicate" in p for p in result.problems)


def test_merge_duplicates_collapses_only_coincident_points():
    X = clustered((3, 3, 3))
    reps, labels = merge_duplicates(X)
    assert reps.shape[0] == 3
    assert labels.shape == (9,)
    assert set(labels.tolist()) == {0, 1, 2}


def test_merge_duplicates_leaves_distinct_points_alone():
    X = I.triangular_lattice(19)
    reps, _ = merge_duplicates(X)
    assert reps.shape[0] == 19


@pytest.mark.parametrize("n", [10, 18, 26])
def test_solver_results_pass_their_own_audit(n):
    res = solve_once(n, seed=0)
    assert res.valid, res.problems
    assert res.distinct_points == n
    assert audit(res.X, res.edge_list, epsilon=1e-9).ok


def test_near_duplicates_are_caught_too():
    """Not just exact overlap: balls a micron apart are the same cheat."""
    X = clustered((4, 4, 4), spread=1e-9, seed=1)
    assert not audit(X).ok
