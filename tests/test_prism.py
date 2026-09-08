"""The stacked-triangle (prism) construction.

Triangle ABC joined to DEF by A-D, B-E, C-F, all nine distances unit,
repeated to tile the plane.
"""

import numpy as np
import pytest

from erdos_unit_distance import init as I
from erdos_unit_distance.counting import audit, count_unit_distances, min_separation

TOL = 1e-9


def closed_form(a: int, b: int) -> int:
    return 3 * (a * b + (a - 1) * b + a * (b - 1) + (a - 1) * (b - 1))


def single_prism(theta=0.4):
    """ABC and DEF = ABC + v, with |v| = 1: the base case of the pattern."""
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    v = np.array([np.cos(theta), np.sin(theta)])
    return np.vstack([tri, tri + v])


def test_a_single_prism_has_nine_unit_distances():
    """Two unit triangles (3 + 3) plus three unit rungs A-D, B-E, C-F."""
    P = single_prism()
    assert P.shape[0] == 6
    assert count_unit_distances(P, TOL) == 9
    assert audit(P, epsilon=TOL).ok


def test_three_unit_rungs_force_a_translation():
    """In the plane, two unit triangles joined by three unit rungs can
    only be translates -- there is no other realisation."""
    P = single_prism()
    offsets = P[3:] - P[:3]
    assert np.allclose(offsets, offsets[0])
    assert np.linalg.norm(offsets[0]) == pytest.approx(1.0)


def test_rungs_parallel_to_a_triangle_edge_collapse_the_prism():
    """The one direction that fails: v equal to an edge vector makes D
    land exactly on B, so the 'extra' distances are duplicate points."""
    P = single_prism(theta=0.0)
    assert not audit(P, epsilon=TOL).ok


@pytest.mark.parametrize("n", [27, 48, 75, 108])
def test_prism_beats_the_triangular_lattice(n):
    assert count_unit_distances(I.prism_lattice(n), TOL) > \
           count_unit_distances(I.triangular_lattice(n), TOL)


@pytest.mark.parametrize("n", [27, 48, 75, 108, 192])
def test_prism_points_are_comfortably_distinct(n):
    """Not merely non-coincident: the default offset keeps real daylight
    between points, so the count cannot be a near-duplicate artefact."""
    P = I.prism_lattice(n)
    result = audit(P, epsilon=TOL)
    assert result.ok, result.problems
    assert result.distinct_points == n
    assert min_separation(P) > 0.5


def test_the_count_is_independent_of_the_triangle_rotation():
    """Only the 60 degrees between v and w matters."""
    counts = {count_unit_distances(I.prism_lattice(75, offset=o), TOL)
              for o in (0.05, 0.3, np.pi / 6, np.pi / 4, np.pi / 2)}
    assert len(counts) == 1


def test_approaches_four_n():
    """3n for the three interleaved lattices, plus n for the triangle
    edges tying them together."""
    ratios = [count_unit_distances(I.prism_lattice(n), TOL) / n
              for n in (48, 108, 300)]
    assert ratios == sorted(ratios)          # monotone toward the limit
    assert ratios[-1] > 3.4
    assert ratios[-1] < 4.0                  # boundary effects keep it under


def test_available_as_an_init():
    X = I.make("prism", 40, np.random.default_rng(0))
    assert X.shape == (40, 2)


# --- the general law: basis Minkowski lattice ------------------------- #

def density(f, n=600):
    P = f(n)
    return count_unit_distances(P, TOL) / n


def test_prism_of_lattices_is_worse_than_prism_of_triangles():
    """Joining two triangular lattice sheets by unit rungs gives a basis
    of just two points and one unit pair, so 3 + 1/2.  A triangle basis
    scores 3 + 3/3 and wins."""
    segment = np.array([[0.0, 0.0], [1.0, 0.0]])
    sheets = density(lambda n: I.lattice_product(n, segment))
    triangles = density(I.prism_lattice)
    assert sheets < triangles


@pytest.mark.parametrize("basis_fn,limit", [
    (lambda: np.zeros((1, 2)), 3.0),                     # plain lattice
    (lambda: np.array([[0.0, 0.0], [1.0, 0.0]]), 3.5),   # prism of lattices
    (I.unit_triangle, 4.0),                              # prism of triangles
    (I.centered_hexagon, 3.0 + 12.0 / 7.0),              # hexagons
])
def test_density_law_predicts_the_ordering(basis_fn, limit):
    """edges/n -> 3 + u(B)/|B|, approached from below on a finite patch."""
    B = basis_fn()
    got = density(lambda n: I.lattice_product(n, B))
    assert got <= limit + 1e-9
    assert got > limit - 1.0          # boundary effects, but the right size


def test_each_doubling_adds_half_an_edge_per_point():
    assert density(I.prism_double) > density(I.prism_lattice)


@pytest.mark.parametrize("fn", [I.prism_lattice, I.prism_double, I.hex_lattice])
def test_lattice_products_have_no_overlapping_points(fn):
    P = fn(400)
    result = audit(P, epsilon=TOL)
    assert result.ok, result.problems
    assert result.distinct_points == 400


def test_tilt_is_what_keeps_the_copies_apart():
    """With no tilt the basis vectors are lattice vectors and copies land
    on top of each other -- a duplicate artefact, not a denser set."""
    flat = I.lattice_product(300, I.unit_triangle(), tilt=0.0)
    assert not audit(flat, epsilon=TOL).ok


@pytest.mark.parametrize("kind", ["prism", "prism_double", "hex_lattice"])
def test_available_as_inits(kind):
    X = I.make(kind, 60, np.random.default_rng(0))
    assert X.shape == (60, 2)


# --- the base lattice term -------------------------------------------- #

def test_rhombic_lattice_has_exactly_one_more_edge_per_point_than_square():
    """A 60-degree rhombus cell has two unit sides plus a unit short
    diagonal; a square cell has two unit sides and a diagonal of sqrt(2).
    So the rhombic lattice carries one more unit distance per point."""
    assert I.density_limit(np.zeros((1, 2)), "triangular") == 3.0
    assert I.density_limit(np.zeros((1, 2)), "square") == 2.0


def test_the_rhombus_short_diagonal_really_is_unit():
    """The geometric fact the whole gap rests on."""
    v = np.array([1.0, 0.0])
    w = np.array([0.5, np.sqrt(3) / 2])
    assert np.linalg.norm(v - w) == pytest.approx(1.0)      # short diagonal
    assert np.linalg.norm(v + w) == pytest.approx(np.sqrt(3.0))
    square_diag = np.linalg.norm(np.array([1.0, 0.0]) - np.array([0.0, 1.0]))
    assert square_diag == pytest.approx(np.sqrt(2.0))       # not unit


@pytest.mark.parametrize("basis_fn", [
    lambda: np.zeros((1, 2)),
    lambda: np.array([[0.0, 0.0], [1.0, 0.0]]),
    I.unit_triangle,
    I.centered_hexagon,
])
def test_the_one_edge_gap_holds_for_every_basis(basis_fn):
    B = basis_fn()
    gap = I.density_limit(B, "triangular") - I.density_limit(B, "square")
    assert gap == pytest.approx(1.0)


@pytest.mark.parametrize("lattice", ["triangular", "square"])
def test_square_base_products_are_valid_point_sets(lattice):
    P = I.lattice_product(400, I.unit_triangle(), lattice=lattice)
    result = audit(P, epsilon=TOL)
    assert result.ok, result.problems
    assert result.distinct_points == 400


def test_rhombic_base_beats_square_base_in_practice():
    for B in (I.unit_triangle(), I.centered_hexagon()):
        tri = count_unit_distances(I.lattice_product(600, B, lattice="triangular"), TOL)
        sq = count_unit_distances(I.lattice_product(600, B, lattice="square"), TOL)
        assert tri > sq


def test_unknown_lattice_is_rejected():
    with pytest.raises(ValueError, match="unknown lattice"):
        I.lattice_product(20, I.unit_triangle(), lattice="hexagonal")
