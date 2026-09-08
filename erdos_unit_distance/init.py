"""Starting configurations for the ball assembly."""

from __future__ import annotations

import numpy as np

# Density of a unit-spacing triangular lattice: 2/sqrt(3) points per unit area.
TRIANGULAR_DENSITY = 2.0 / np.sqrt(3.0)

INITS = ("random", "triangular", "square", "erdos_grid", "hex_flower",
         "hex_minkowski", "prism", "prism_double", "hex_lattice",
         "moser", "circle")


def _closest_to_centre(points: np.ndarray, n: int) -> np.ndarray:
    """Keep the ``n`` points nearest the centroid, so the patch stays compact."""
    c = points.mean(axis=0)
    order = np.argsort(((points - c) ** 2).sum(axis=1), kind="stable")
    return points[order[:n]]


def random_disc(n: int, rng: np.random.Generator, density: float = TRIANGULAR_DENSITY) -> np.ndarray:
    """Uniform points in a disc sized so the mean spacing is about 1."""
    radius = np.sqrt(n / (np.pi * density))
    r = radius * np.sqrt(rng.random(n))
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


def triangular_lattice(n: int) -> np.ndarray:
    """Compact patch of the unit-spacing triangular lattice.

    The classic easy baseline: every interior ball has 6 neighbours at
    distance 1, giving 3n - O(sqrt(n)) unit distances.
    """
    k = int(np.ceil(np.sqrt(n))) + 3
    a, b = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1), indexing="ij")
    a, b = a.ravel(), b.ravel()
    pts = np.stack([a + 0.5 * b, b * np.sqrt(3.0) / 2.0], axis=1)
    return _closest_to_centre(pts, n)


def square_lattice(n: int) -> np.ndarray:
    """Compact patch of the unit-spacing square lattice (2n - O(sqrt(n)) edges)."""
    k = int(np.ceil(np.sqrt(n))) + 3
    a, b = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1), indexing="ij")
    pts = np.stack([a.ravel(), b.ravel()], axis=1).astype(np.float64)
    return _closest_to_centre(pts, n)


def erdos_grid(n: int) -> np.ndarray:
    """Erdos' own construction: a sqrt(n) x sqrt(n) integer grid, rescaled.

    Pick the squared distance ``r`` realised by the most pairs of grid
    points and scale the grid by ``1/sqrt(r)``, turning every one of those
    pairs into a unit distance.  Because the number of representations of
    an integer as a sum of two squares spikes on integers with many prime
    factors of the form 4k+1, this beats any lattice with a fixed
    neighbourhood and gives the n^(1+c/log log n) lower bound.
    """
    k = int(np.ceil(np.sqrt(n)))
    a, b = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    pts = np.stack([a.ravel(), b.ravel()], axis=1).astype(np.float64)
    pts = _closest_to_centre(pts, n)

    diff = pts[:, None, :] - pts[None, :, :]
    sq = np.rint((diff ** 2).sum(axis=2)).astype(np.int64)
    iu = np.triu_indices(pts.shape[0], k=1)
    vals = sq[iu]
    vals = vals[vals > 0]
    if vals.size == 0:
        return pts
    best_r = np.bincount(vals).argmax()
    return pts / np.sqrt(float(best_r))


def hex_flower(n: int) -> np.ndarray:
    """Rings of a hexagonal packing: the n = 7 case is the known optimum (12).

    The 7-point case is the *centered hexagon* -- a hub plus six unit
    vectors, the wheel graph W6, with 6 spokes and 6 rim edges.  Tiling it
    is just the triangular lattice, which is why this defers to it.
    """
    return triangular_lattice(n)


def _dedupe(P: np.ndarray, tol: float = 1e-7) -> np.ndarray:
    """Drop coincident points, which Minkowski sums produce in bulk."""
    _, keep = np.unique(np.round(P / tol).astype(np.int64), axis=0, return_index=True)
    return P[np.sort(keep)]


def centered_hexagon(theta: float = 0.0) -> np.ndarray:
    """Hub plus six unit vectors: 7 points, 12 unit distances, optimal."""
    a = theta + np.arange(6) * np.pi / 3.0
    return np.vstack([[0.0, 0.0], np.stack([np.cos(a), np.sin(a)], axis=1)])


def hex_minkowski(n: int, angles=(0.0, 0.3, 0.7, 1.1)) -> np.ndarray:
    """Minkowski sums of *rotated* centered hexagons.

    Summing k copies at generic angles gives 7^k distinct points carrying
    exactly ``12 * k * 7**(k-1)`` unit distances -- two points are a unit
    apart whenever they differ in a single copy by one of its spokes or
    rim steps.  That is ``(12/7) * log_7(n) * n``, so it grows like
    ``n log n`` and pulls away from the triangular lattice's 3n.

    Rotating matters: summing unrotated copies just regenerates the
    triangular lattice, because the six hexagon vectors generate it.
    """
    P = np.zeros((1, 2))
    for k in range(len(angles)):
        if P.shape[0] >= n:
            break
        P = _dedupe((P[:, None, :] + centered_hexagon(angles[k])[None, :, :]).reshape(-1, 2))
    return _closest_to_centre(P, n)


def prism_lattice(n: int, offset: float = np.pi / 6.0) -> np.ndarray:
    """Unit triangles stacked in prisms, tiled over the plane.

    Take a unit triangle ABC and join it to a second triangle DEF with
    A-D, B-E, C-F all unit.  In the plane that forces DEF to be a
    *translate* of ABC by a unit vector, so the whole family is

        TRIANGLE + i*v + j*w,   |v| = |w| = 1

    and the only real freedom is the angle between v and w.  At exactly
    60 degrees ``v - w`` is a unit vector too, so the diagonal neighbours
    are joined as well, and the count jumps from 3n to

        3 * [ab + (a-1)b + a(b-1) + (a-1)(b-1)]  ->  4n - O(sqrt(n))

    which beats the triangular lattice.  Seen another way this is the
    Minkowski sum of a unit triangle with a unit triangular lattice: three
    interleaved lattices (3n edges between them) plus the triangle edges
    that tie them together (another n).

    ``offset`` rotates the triangle relative to v and w.  It does not
    change the count at all, but it does change how close distinct points
    come to each other, so the default is a value with comfortable
    separation rather than one that merely avoids exact collisions.
    """
    return lattice_product(n, unit_triangle(), tilt=offset)


#: Unit cell vectors, and the resulting unit distances per lattice point.
#:
#: The triangular lattice is a lattice of 60-degree rhombi.  Such a rhombus
#: has unit sides *and* a unit short diagonal (the long one is sqrt(3)), so
#: each cell contributes two sides plus that diagonal.  A square cell has
#: only its two sides -- its diagonal is sqrt(2), not 1.  Hence exactly one
#: more unit distance per point on the rhombic lattice: 3 against 2.
LATTICES = {
    "triangular": (np.array([1.0, 0.0]), np.array([0.5, np.sqrt(3.0) / 2.0]), 3),
    "square": (np.array([1.0, 0.0]), np.array([0.0, 1.0]), 2),
}


def lattice_product(n: int, basis: np.ndarray, tilt: float = 0.4,
                    lattice: str = "triangular") -> np.ndarray:
    """Minkowski sum of a small point set with a unit lattice.

    This generalises the prism.  Place a copy of ``basis`` at every lattice
    site; per site the result has

        z/2 * |B|          edges inside each basis point's own lattice copy
      + u(B)               edges between copies, one per unit pair in B

    where ``z/2`` is the lattice's own unit distances per point.  So the
    asymptotic density is

        edges / n  ->  z/2 + u(B) / |B|

    Both terms are levers.  The base lattice sets the constant: a rhombic
    (triangular) lattice gives 3, a square one only 2, because the 60-degree
    rhombus contributes a unit short diagonal that the square cell has no
    equivalent of.

    The basis sets the rest, and says what to feed it: not another lattice,
    but the *densest small unit-distance set you have*.  A segment (a "prism
    of two lattices") scores u/|B| = 1/2.  A unit triangle scores 1, giving
    4n on the rhombic lattice.  The centered hexagon scores 12/7.  A flower
    sum scores 168/49 -- and since that ratio itself grows like the log of
    the basis size, so does this.

    ``tilt`` rotates the basis off the lattice directions.  Without it the
    basis vectors coincide with lattice vectors and copies land on top of
    each other, which is a duplicate-point artefact rather than a result.
    """
    if lattice not in LATTICES:
        raise ValueError(f"unknown lattice {lattice!r}; expected one of {tuple(LATTICES)}")
    basis = np.asarray(basis, dtype=np.float64).reshape(-1, 2)
    c, s_ = np.cos(tilt), np.sin(tilt)
    basis = basis @ np.array([[c, s_], [-s_, c]])

    v, w, _ = LATTICES[lattice]
    k = int(np.ceil(np.sqrt(max(n / max(len(basis), 1), 1.0)))) + 2
    sites = np.array([i * v + j * w for i in range(-k, k + 1) for j in range(-k, k + 1)])
    pts = (sites[:, None, :] + basis[None, :, :]).reshape(-1, 2)
    return _closest_to_centre(pts, n)


def density_limit(basis: np.ndarray, lattice: str = "triangular") -> float:
    """The predicted ``edges / n`` for ``basis`` on ``lattice``."""
    from .counting import count_unit_distances

    basis = np.asarray(basis, dtype=np.float64).reshape(-1, 2)
    per_point = LATTICES[lattice][2]
    u = count_unit_distances(basis, 1e-9) if len(basis) > 1 else 0
    return per_point + u / len(basis)


def unit_triangle() -> np.ndarray:
    """A single unit equilateral triangle."""
    return np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])


def prism_double(n: int, offset: float = 0.9) -> np.ndarray:
    """The prism doubled: a triangle, plus the same triangle a unit away.

    Each such doubling adds one more unit pair per basis point, and so
    lifts the density by exactly 1/2: 4n, then 4.5n, then 5n, ...
    """
    tri = unit_triangle()
    u = np.array([np.cos(offset), np.sin(offset)])
    return lattice_product(n, np.vstack([tri, tri + u]))


def hex_lattice(n: int) -> np.ndarray:
    """Centered hexagons on the triangular lattice: 3 + 12/7 = 4.71n."""
    return lattice_product(n, centered_hexagon())


def moser_spindle() -> np.ndarray:
    """The Moser spindle: 7 points, 11 unit distances, chromatic number 4.

    Two rhombi built from unit equilateral triangles, hinged at a shared
    apex and rotated until their far tips are themselves a unit apart.
    """
    s3 = np.sqrt(3.0)

    def rhombus(theta):
        # Apex at the origin; two unit triangles glued along a common edge.
        base = np.array([[s3 / 2.0, 0.5], [s3 / 2.0, -0.5], [s3, 0.0]])
        c, s = np.cos(theta), np.sin(theta)
        return base @ np.array([[c, s], [-s, c]])

    # Choose the hinge angle so the two far tips sit exactly one unit apart.
    half = np.arcsin(0.5 / s3)
    top, bot = rhombus(half), rhombus(-half)
    return np.vstack([np.zeros((1, 2)), top, bot])


def circle(n: int) -> np.ndarray:
    """n points on a unit-radius circle (every chord to the centre is 1)."""
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([np.cos(theta), np.sin(theta)], axis=1)


def make(kind: str, n: int, rng: np.random.Generator, jitter: float = 0.0) -> np.ndarray:
    """Build a starting configuration of ``n`` balls."""
    if kind == "random":
        X = random_disc(n, rng)
    elif kind == "triangular":
        X = triangular_lattice(n)
    elif kind == "square":
        X = square_lattice(n)
    elif kind == "erdos_grid":
        X = erdos_grid(n)
    elif kind == "hex_flower":
        X = hex_flower(n)
    elif kind == "hex_minkowski":
        X = hex_minkowski(n)
    elif kind == "prism":
        X = prism_lattice(n)
    elif kind == "prism_double":
        X = prism_double(n)
    elif kind == "hex_lattice":
        X = hex_lattice(n)
    elif kind == "circle":
        X = circle(n)
    elif kind == "moser":
        X = moser_spindle()
        if n != X.shape[0]:
            extra = random_disc(max(n - X.shape[0], 0), rng)
            X = np.vstack([X, extra])[:n]
    else:
        raise ValueError(f"unknown init {kind!r}; expected one of {INITS}")

    if jitter:
        X = X + rng.normal(scale=jitter, size=X.shape)
    return X - X.mean(axis=0, keepdims=True)
