"""Are honeycomb-family constructions good unit-distance candidates?

The centered hexagon -- six unit vectors around a hub, the wheel graph W6
-- has 12 unit distances on 7 points, which is the proven optimum u(7).
This asks whether building bigger point sets out of it keeps paying off.

Four families are compared against the two classical baselines:

  centered_hex   tile the plane with it  =  the triangular lattice
  honeycomb      the 3-regular graphene lattice
  hex_minkowski  sums of k rotated copies of the flower
  hex_of_hex     flowers placed at the vertices of a larger flower
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from erdos_unit_distance.counting import count_unit_distances, min_separation
from erdos_unit_distance.init import (eisenstein_grid, erdos_grid,
                                      triangular_lattice)

TOL = 1e-9


def dedupe(P: np.ndarray, tol: float = 1e-7) -> np.ndarray:
    """Drop duplicate points, which Minkowski sums produce in bulk."""
    keys = np.round(P / tol).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    return P[np.sort(keep)]


def compact(P: np.ndarray, n: int) -> np.ndarray:
    """The n points of P nearest its centroid."""
    c = P.mean(axis=0)
    return P[np.argsort(((P - c) ** 2).sum(axis=1), kind="stable")[:n]]


def flower(theta: float = 0.0) -> np.ndarray:
    """Centered hexagon: hub plus six unit vectors.  12 unit distances."""
    a = theta + np.arange(6) * np.pi / 3.0
    return np.vstack([[0.0, 0.0], np.stack([np.cos(a), np.sin(a)], axis=1)])


def minkowski(*sets: np.ndarray) -> np.ndarray:
    out = np.zeros((1, 2))
    for S in sets:
        out = dedupe((out[:, None, :] + S[None, :, :]).reshape(-1, 2))
    return out


def honeycomb(n: int) -> np.ndarray:
    """The 3-regular honeycomb (graphene) lattice, unit bond length."""
    a1 = np.array([1.5, np.sqrt(3) / 2.0])
    a2 = np.array([1.5, -np.sqrt(3) / 2.0])
    basis = np.array([[0.0, 0.0], [1.0, 0.0]])
    k = int(np.ceil(np.sqrt(n))) + 3
    cells = np.array([i * a1 + j * a2 for i in range(-k, k + 1) for j in range(-k, k + 1)])
    return compact(dedupe((cells[:, None, :] + basis[None, :, :]).reshape(-1, 2)), n)


def hex_minkowski(n: int, thetas) -> np.ndarray:
    """Sum of rotated flowers.  All-zero rotations collapse to the lattice."""
    return compact(minkowski(*[flower(t) for t in thetas]), n)


def hex_of_hex(n: int, scale: float = 1.0) -> np.ndarray:
    """A flower of flowers: copies placed at a scaled flower's vertices."""
    return compact(dedupe((scale * flower()[:, None, :] + flower()[None, :, :]).reshape(-1, 2)), n)


def report(name: str, P: np.ndarray) -> tuple[int, float]:
    n = P.shape[0]
    c = count_unit_distances(P, TOL)
    print(f"  {name:<34} n={n:<5d} unit distances={c:<6d} "
          f"({c / n:.2f}n)  min sep={min_separation(P):.3f}")
    return c, c / n


def main():
    print("baselines")
    for n in (19, 37, 61):
        report(f"triangular lattice (n={n})", triangular_lattice(n))
    for n in (19, 37, 61):
        report(f"rescaled integer grid (n={n})", erdos_grid(n))

    print("\ncentered hexagon and its tilings")
    report("centered hexagon W6", flower())
    report("honeycomb / graphene (n=61)", honeycomb(61))

    print("\nMinkowski sums of rotated flowers")
    # theta = 0 twice just regenerates the triangular lattice; the point of
    # rotating is to create distances the lattice does not contain.
    for thetas in [(0.0, 0.0), (0.0, 0.3), (0.0, 0.55), (0.0, np.pi / 6),
                   (0.0, 0.0, 0.0), (0.0, 0.3, 0.7)]:
        P = dedupe(minkowski(*[flower(t) for t in thetas]))
        label = "+".join(f"{t:.2f}" for t in thetas)
        report(f"flowers at {label}", P)

    print("\nflower of flowers, varying the outer scale")
    for scale in (1.0, np.sqrt(3.0), 2.0, 2.5, 3.0):
        report(f"outer scale {scale:.3f}", hex_of_hex(200, scale))

    print("\nhead to head at matched n")
    print(f"  {'n':>5}  {'triangular':>11}  {'grid':>6}  {'eisenstein':>11}  "
          f"{'honeycomb':>10}  {'minkowski':>10}")
    for n in (19, 37, 61, 91, 127):
        tri = count_unit_distances(triangular_lattice(n), TOL)
        grid = count_unit_distances(erdos_grid(n), TOL)
        hc = count_unit_distances(honeycomb(n), TOL)
        best_mk = max(
            count_unit_distances(hex_minkowski(n, th), TOL)
            for th in [(0.0, 0.3), (0.0, 0.55), (0.0, np.pi / 6), (0.0, 0.3, 0.7)])
        eis = count_unit_distances(eisenstein_grid(n), TOL)
        print(f"  {n:>5}  {tri:>11}  {grid:>6}  {eis:>11}  {hc:>10}  {best_mk:>10}")


if __name__ == "__main__":
    main()
