"""Does the Erdos rescaling trick help any of the other constructions?

Erdos' grid works in two steps that are worth separating:

1. Build a point set whose pairwise squared distances are *integers*.
2. Find the value realised by the most pairs and scale by 1/sqrt(it), so
   those pairs all become unit distances.

Step 2 is generic -- it applies to any point set at all.  Step 1 is the
arithmetic part, and it is what actually does the work: integers have
wildly uneven numbers of representations as a sum of two squares, so the
histogram of squared distances has a tall spike to aim at.

So the question is which of our constructions have a spike worth
rescaling to.  This measures the distance histogram of each one and
compares the best available rescaling against the unit count it already
had.
"""

from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from erdos_unit_distance import init as I
from erdos_unit_distance.counting import audit, count_unit_distances

TOL = 1e-9
#: Squared distances are bucketed at this resolution before counting.
BUCKET = 1e-7


def squared_distances(P: np.ndarray, block: int = 512) -> np.ndarray:
    """All n(n-1)/2 squared pair distances."""
    n = P.shape[0]
    out = []
    for lo in range(0, n, block):
        hi = min(lo + block, n)
        d = ((P[lo:hi, None, :] - P[None, :, :]) ** 2).sum(axis=2)
        keep = np.arange(lo, hi)[:, None] < np.arange(n)[None, :]
        out.append(d[keep])
    return np.concatenate(out)


def popular_distance(P: np.ndarray):
    """Return ``(multiplicity, squared_distance)`` of the commonest pair gap."""
    sq = squared_distances(P)
    sq = sq[sq > BUCKET]
    if sq.size == 0:
        return 0, 0.0
    counts = Counter(np.round(sq / BUCKET).astype(np.int64).tolist())
    key, mult = counts.most_common(1)[0]
    return mult, key * BUCKET


def rescale_to_popular(P: np.ndarray) -> np.ndarray:
    """Scale so the most frequently realised distance becomes exactly 1."""
    mult, sq = popular_distance(P)
    if mult == 0 or sq <= 0.0:
        return P
    return P / np.sqrt(sq)


def spectrum(P: np.ndarray, top: int = 5):
    """The most popular few squared distances, with their multiplicities."""
    sq = squared_distances(P)
    sq = sq[sq > BUCKET]
    counts = Counter(np.round(sq / BUCKET).astype(np.int64).tolist())
    return [(m, k * BUCKET) for k, m in counts.most_common(top)]


FAMILIES = [
    ("square lattice", I.square_lattice),
    ("triangular lattice", I.triangular_lattice),
    ("prism of triangles", I.prism_lattice),
    ("prism doubled", I.prism_double),
    ("centered hexagon x lattice", I.hex_lattice),
    ("rotated flower sums", I.hex_minkowski),
    ("Erdos rescaled grid", I.erdos_grid),
]


def main():
    for n in (300, 900):
        print(f"\n{'=' * 78}\nn = {n}\n{'=' * 78}")
        print(f"{'construction':<28}{'as built':>10}{'rescaled':>10}"
              f"{'gain':>8}   {'scale':>8}  audit")
        print("-" * 78)
        for name, fn in FAMILIES:
            P = fn(n)
            before = count_unit_distances(P, TOL)
            Q = rescale_to_popular(P)
            after = count_unit_distances(Q, TOL)
            _, sq = popular_distance(P)
            a = audit(Q, epsilon=TOL)
            gain = f"{after / max(before, 1):.2f}x"
            print(f"{name:<28}{before:>10}{after:>10}{gain:>8}   "
                  f"{np.sqrt(sq):>8.4f}  {'PASS' if a.ok else 'FAIL'}")

    print(f"\n{'=' * 78}")
    print("distance spectrum: how tall is the spike to aim at?")
    print(f"{'=' * 78}")
    for name, fn in FAMILIES:
        P = fn(900)
        rows = spectrum(P)
        pretty = "  ".join(f"{m}@{np.sqrt(s):.3f}" for m, s in rows)
        print(f"  {name:<28} {pretty}")

    print("\nwhy generic rotations cannot benefit: how many *distinct*")
    print("pair distances does each construction realise?")
    for name, fn in FAMILIES:
        P = fn(900)
        sq = squared_distances(P)
        distinct = len(set(np.round(sq / BUCKET).astype(np.int64).tolist()))
        pairs = sq.size
        print(f"  {name:<28} {distinct:>7} distinct out of {pairs} pairs "
              f"({pairs / distinct:>6.1f} pairs per distance on average)")


if __name__ == "__main__":
    main()
