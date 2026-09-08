"""Reference values to score a run against.

``MAX_UNIT_DISTANCES`` holds u(n), the maximum number of unit distances
among n points in the plane, for the small n where it has actually been
determined by exhaustive search (Schade, 1993).  These are the honest
targets; anything larger would be a counterexample rather than a result,
so treat a run that reports more as a bug in the counting tolerance.

Cross-check against OEIS A186705 before relying on these.
"""

from __future__ import annotations

# n -> u(n), for n = 1..14.
MAX_UNIT_DISTANCES = {
    1: 0, 2: 1, 3: 3, 4: 5, 5: 7, 6: 9, 7: 12,
    8: 14, 9: 18, 10: 20, 11: 23, 12: 27, 13: 30, 14: 33,
}


def known_optimum(n: int):
    """Return u(n) if it is known exactly, else ``None``."""
    return MAX_UNIT_DISTANCES.get(n)


def triangular_baseline(n: int) -> int:
    """Unit distances in a compact triangular-lattice patch (~3n - O(sqrt n))."""
    from .counting import count_unit_distances
    from .init import triangular_lattice

    return count_unit_distances(triangular_lattice(n), 1e-9)


def erdos_grid_baseline(n: int) -> int:
    """Unit distances in the rescaled sqrt(n) x sqrt(n) integer grid."""
    from .counting import count_unit_distances
    from .init import erdos_grid

    return count_unit_distances(erdos_grid(n), 1e-9)


#: Never let the hard core exclude a configuration this good, but never
#: relax it past the point where near-duplicate balls become a viable
#: cheat.  0.4 sits just under the Moser spindle (0.457) and the rescaled
#: integer grid (0.447), the tightest genuinely good small configurations.
R_MIN_CAP = 0.4
R_MIN_FLOOR = 0.10


def default_r_min(n: int) -> float:
    """Hard-core radius that cannot exclude the best construction we know.

    The balls must stay distinct, and a core that is too small lets the
    relaxation approximate the degenerate "stack duplicate points a unit
    apart" configuration, which scores ~n^2/3 spurious unit distances.
    A core that is too large excludes real answers: the rescaled integer
    grid puts genuine points 1/sqrt(r) apart, and that shrinks with n.

    So the default tracks the grid's own minimum separation at this n,
    with a little headroom, clamped to a sane range.
    """
    from .counting import min_separation
    from .init import erdos_grid

    sep = min_separation(erdos_grid(n)) if n >= 2 else float("inf")
    return float(min(R_MIN_CAP, max(R_MIN_FLOOR, 0.9 * sep)))


def summarise(n: int, achieved: int) -> str:
    """One-line comparison of a result against the reference values."""
    bits = [f"n={n}  edges={achieved}"]
    opt = known_optimum(n)
    if opt is not None:
        bits.append(f"known u(n)={opt} ({achieved - opt:+d})")
    tri = triangular_baseline(n)
    grid = erdos_grid_baseline(n)
    bits.append(f"triangular={tri} ({achieved - tri:+d})")
    bits.append(f"erdos_grid={grid} ({achieved - grid:+d})")
    bits.append(f"ratio={achieved / max(n, 1):.2f}n")
    return "  |  ".join(bits)
