"""Empirical growth rates for every construction, against the theory.

For each family this measures u(n) over a range of sizes and reports the
local log-log slope

    d log(edges) / d log(n)

which is the exponent a in edges ~ n^a.  A slope pinned at 1.0 means the
family is Theta(n) and only the constant differs; a slope drifting above
1 means genuinely superlinear growth.

Density (edges / n) is the more discriminating readout at these sizes,
because a Theta(n log n) family only shows up as a slope of about
1 + 1/log(n), which is very close to 1 for any n one can actually build.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from erdos_unit_distance import init as I
from erdos_unit_distance.counting import audit, count_unit_distances

TOL = 1e-9

FAMILIES = [
    ("square lattice", I.square_lattice, "2n - O(sqrt n)"),
    ("triangular lattice", I.triangular_lattice, "3n - O(sqrt n)"),
    ("prism: triangle x lattice", I.prism_lattice, "4n - O(sqrt n)"),
    ("prism doubled up+down", I.prism_double, "4.5n - O(sqrt n)"),
    ("centered hexagon x lattice", I.hex_lattice, "(3+12/7)n - O(sqrt n)"),
    ("Erdos rescaled grid", I.erdos_grid, "n^(1+c/log log n)"),
]

SIZES = (100, 250, 600, 1500, 3000)


def measure(fn, sizes):
    rows = []
    for n in sizes:
        P = fn(n)
        e = count_unit_distances(P, TOL)
        rows.append((n, e))
    return rows


def slopes(rows):
    out = [float("nan")]
    for (n0, e0), (n1, e1) in zip(rows, rows[1:]):
        if e0 > 0 and e1 > 0:
            out.append(np.log(e1 / e0) / np.log(n1 / n0))
        else:
            out.append(float("nan"))
    return out


def main():
    print(f"sizes: {SIZES}\n")
    header = f"{'construction':<28}" + "".join(f"{n:>9}" for n in SIZES)
    print(header)
    print("-" * len(header))

    table = {}
    for name, fn, formula in FAMILIES:
        rows = measure(fn, SIZES)
        table[name] = (rows, formula)
        print(f"{name:<28}" + "".join(f"{e:>9d}" for _, e in rows))

    print(f"\n{'density (edges / n)':<28}" + "".join(f"{n:>9}" for n in SIZES))
    print("-" * len(header))
    for name, (rows, _) in table.items():
        print(f"{name:<28}" + "".join(f"{e / n:>9.2f}" for n, e in rows))

    print(f"\n{'local log-log slope':<28}" + "".join(f"{n:>9}" for n in SIZES))
    print("-" * len(header))
    for name, (rows, _) in table.items():
        s = slopes(rows)
        cells = "".join("      ---" if np.isnan(v) else f"{v:>9.3f}" for v in s)
        print(f"{name:<28}{cells}")

    print("\nclosed forms")
    for name, (_, formula) in table.items():
        print(f"  {name:<28} {formula}")

    print("\nsanity: every configuration audited for coincident points")
    for name, fn, _ in FAMILIES:
        a = audit(fn(600), epsilon=TOL)
        print(f"  {name:<28} {'PASS' if a.ok else 'FAIL'}  "
              f"({a.distinct_points}/600 distinct)")

    # Flower sums are only structurally clean at n = 7^k.  Truncating one
    # to an arbitrary n cuts through the structure and the count collapses,
    # so measuring them on the shared grid above would be misleading.
    print("\nrotated flower sums, at their natural sizes n = 7^k")
    print(f"  {'n':>7} {'edges':>9} {'edges/n':>9} {'slope':>8}   vs 12k*7^(k-1)")
    prev = None
    for k in range(1, 6):
        n = 7 ** k
        e = count_unit_distances(I.hex_minkowski(n), TOL)
        slope = "" if prev is None else f"{np.log(e / prev[1]) / np.log(n / prev[0]):.3f}"
        exact = "exact" if e == 12 * k * 7 ** (k - 1) else "MISMATCH"
        print(f"  {n:>7} {e:>9} {e / n:>9.2f} {slope:>8}   {exact}")
        prev = (n, e)

    print("\nwhere the asymptotically better construction actually overtakes")
    for k in range(2, 6):
        n = 7 ** k
        flower = count_unit_distances(I.hex_minkowski(n), TOL)
        grid = count_unit_distances(I.erdos_grid(n), TOL)
        lead = "flower sums" if flower > grid else "Erdos grid"
        print(f"  n = {n:>5}: flower sums {flower:>7}, grid {grid:>7}"
              f"   -> {lead} ahead")


if __name__ == "__main__":
    main()
