"""The stacked-triangle (prism) pattern, measured.

Triangle ABC joined to triangle DEF by A-D, B-E, C-F, with all nine
distances equal to 1; then GHI joined to DEF the same way, and so on.

The realisation in the plane is forced and simple: if every rung
A-D, B-E, C-F is to be unit *and* DEF is a unit triangle, then DEF must be
a **translate** of ABC by some unit vector v.  So a chain of k triangles
is T, T+v, T+2v, ..., and tiling the plane means translating by two unit
vectors v and w.

That leaves only the angles of v and w free, which is what this scans.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from erdos_unit_distance.counting import audit, count_unit_distances, min_separation
from erdos_unit_distance.init import erdos_grid, hex_minkowski, triangular_lattice

TOL = 1e-9
TRIANGLE = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2.0]])


def unit_vec(theta: float) -> np.ndarray:
    return np.array([np.cos(theta), np.sin(theta)])


def prism_strip(k: int, theta: float) -> np.ndarray:
    """k unit triangles, each the previous one translated by a unit vector."""
    v = unit_vec(theta)
    return np.vstack([TRIANGLE + j * v for j in range(k)])


def prism_lattice(a: int, b: int, theta_v: float, theta_w: float) -> np.ndarray:
    """Triangles translated on a lattice spanned by two unit vectors."""
    v, w = unit_vec(theta_v), unit_vec(theta_w)
    return np.vstack([TRIANGLE + i * v + j * w
                      for i in range(a) for j in range(b)])


def score(P: np.ndarray):
    a = audit(P, epsilon=TOL)
    return a.edges_claimed, a.ok, min_separation(P)


def main():
    print("chain of k triangles, scanning the rung direction")
    print(f"  {'k':>3} {'n':>4}  {'best theta':>10} {'edges':>6} {'per point':>10}  audit")
    for k in (2, 3, 5, 10, 20):
        best = None
        for theta in np.linspace(0.0, np.pi, 721):
            P = prism_strip(k, theta)
            c, ok, sep = score(P)
            if ok and (best is None or c > best[0]):
                best = (c, theta, sep)
        c, theta, sep = best
        n = 3 * k
        print(f"  {k:>3} {n:>4}  {np.degrees(theta):>9.2f}d {c:>6} "
              f"{c / n:>9.2f}n  min sep {sep:.3f}")

    print("\nthe forced degenerate angles (rungs parallel to a triangle edge)")
    for deg in (0.0, 60.0, 120.0):
        P = prism_strip(3, np.radians(deg))
        a = audit(P, epsilon=TOL)
        print(f"  theta={deg:>5.1f}d  claims {a.edges_claimed} edges but "
              f"{a.n - a.distinct_points} points coincide -> "
              f"{'PASS' if a.ok else 'REJECTED'}")

    print("\ntiling the plane: triangles on a lattice of two unit vectors")
    print(f"  {'grid':>7} {'n':>5}  {'v,w angles':>16} {'edges':>6} {'per point':>10}  audit")
    for a_, b_ in ((2, 2), (3, 3), (4, 4), (5, 5), (6, 6)):
        best = None
        for tv in np.linspace(0.0, np.pi, 61):
            for tw in np.linspace(0.0, np.pi, 61):
                if abs(tv - tw) < 1e-9:
                    continue
                P = prism_lattice(a_, b_, tv, tw)
                c, ok, sep = score(P)
                if ok and (best is None or c > best[0]):
                    best = (c, tv, tw, sep)
        if best is None:
            continue
        c, tv, tw, sep = best
        n = 3 * a_ * b_
        print(f"  {a_}x{b_:<5} {n:>5}  {np.degrees(tv):>6.1f},{np.degrees(tw):>6.1f}d "
              f"{c:>6} {c / n:>9.2f}n  min sep {sep:.3f}")

    print("\nhead to head at matched n")
    print(f"  {'n':>5} {'prism tiling':>13} {'triangular':>11} {'grid':>6} {'flower sums':>12}")
    for a_, b_ in ((3, 3), (4, 4), (5, 5)):
        n = 3 * a_ * b_
        best = 0
        for tv in np.linspace(0.0, np.pi, 61):
            for tw in np.linspace(0.0, np.pi, 61):
                if abs(tv - tw) < 1e-9:
                    continue
                P = prism_lattice(a_, b_, tv, tw)
                c, ok, _ = score(P)
                if ok:
                    best = max(best, c)
        print(f"  {n:>5} {best:>13} "
              f"{count_unit_distances(triangular_lattice(n), TOL):>11} "
              f"{count_unit_distances(erdos_grid(n), TOL):>6} "
              f"{count_unit_distances(hex_minkowski(n), TOL):>12}")


if __name__ == "__main__":
    main()
