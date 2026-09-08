"""Turning a relaxed spring assembly into a unit-distance graph.

A spring is promoted to an **edge** when its length is within ``epsilon``
of 1.  Everything the optimiser is ultimately scored on comes from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_BLOCK = 512

#: Two balls closer than this are treated as the same point.  The Erdos
#: problem is about n *distinct* points, and coincident points are the
#: cheapest way to fake a high count: three coincident clusters a unit
#: apart give ~n^2/3 "unit distances" without being a point set at all.
DUP_TOL = 1e-6


def unit_edges(X: np.ndarray, epsilon: float) -> np.ndarray:
    """Indices ``(i, j)``, ``i < j``, of every spring within ``epsilon`` of unit.

    Returns an ``(m, 2)`` int array, lexicographically ordered.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    rows, cols = [], []
    for lo in range(0, n, _BLOCK):
        hi = min(lo + _BLOCK, n)
        diff = X[lo:hi, None, :] - X[None, :, :]
        d = np.sqrt(np.einsum("bnk,bnk->bn", diff, diff))
        hit = np.abs(d - 1.0) <= epsilon
        # keep only the i < j half
        hit &= np.arange(lo, hi)[:, None] < np.arange(n)[None, :]
        r, c = np.nonzero(hit)
        rows.append(r + lo)
        cols.append(c)
    if not rows:
        return np.zeros((0, 2), dtype=np.int64)
    return np.stack([np.concatenate(rows), np.concatenate(cols)], axis=1)


def count_unit_distances(X: np.ndarray, epsilon: float) -> int:
    """Number of springs within ``epsilon`` of unit length."""
    return int(unit_edges(X, epsilon).shape[0])


def edge_lengths(X: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Length of each listed edge."""
    if edges.shape[0] == 0:
        return np.zeros(0)
    v = X[edges[:, 0]] - X[edges[:, 1]]
    return np.sqrt((v * v).sum(axis=1))


def min_separation(X: np.ndarray) -> float:
    """Smallest distance between two distinct balls (0 if they coincide)."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < 2:
        return float("inf")
    best = np.inf
    for lo in range(0, n, _BLOCK):
        hi = min(lo + _BLOCK, n)
        diff = X[lo:hi, None, :] - X[None, :, :]
        d = np.sqrt(np.einsum("bnk,bnk->bn", diff, diff))
        d[np.arange(hi - lo), np.arange(lo, hi)] = np.inf
        best = min(best, float(d.min()))
    return best


def degree_sequence(n: int, edges: np.ndarray) -> np.ndarray:
    """Unit-distance degree of every ball."""
    deg = np.zeros(n, dtype=np.int64)
    if edges.shape[0]:
        np.add.at(deg, edges[:, 0], 1)
        np.add.at(deg, edges[:, 1], 1)
    return deg


def report(X: np.ndarray, epsilons=(1e-1, 1e-2, 1e-3, 1e-6)) -> dict:
    """Edge counts at a ladder of tolerances, plus basic geometry health."""
    counts = {f"{eps:g}": count_unit_distances(X, eps) for eps in epsilons}
    tight = unit_edges(X, min(epsilons))
    lengths = edge_lengths(X, tight)
    return {
        "n": int(X.shape[0]),
        "counts": counts,
        "min_separation": min_separation(X),
        "max_edge_error": float(np.abs(lengths - 1.0).max()) if lengths.size else 0.0,
        "max_degree": int(degree_sequence(X.shape[0], tight).max()) if X.shape[0] else 0,
    }


def merge_duplicates(X: np.ndarray, tol: float = DUP_TOL):
    """Collapse balls closer than ``tol`` into single points.

    Returns ``(representatives, labels)`` where ``labels[i]`` is the index
    into ``representatives`` for original ball ``i``.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for lo in range(0, n, _BLOCK):
        hi = min(lo + _BLOCK, n)
        diff = X[lo:hi, None, :] - X[None, :, :]
        d = np.sqrt(np.einsum("bnk,bnk->bn", diff, diff))
        hit = (d <= tol) & (np.arange(lo, hi)[:, None] < np.arange(n)[None, :])
        for r, c in zip(*np.nonzero(hit)):
            a, b = find(int(r) + lo), find(int(c))
            if a != b:
                parent[max(a, b)] = min(a, b)

    roots = np.array([find(i) for i in range(n)])
    uniq, labels = np.unique(roots, return_inverse=True)
    return X[uniq], labels


@dataclass
class Audit:
    """Result of checking that a reported count is honest."""

    ok: bool
    n: int
    distinct_points: int
    min_separation: float
    edges_claimed: int
    edges_recounted: int
    edges_among_distinct_points: int
    max_edge_error: float
    problems: list = field(default_factory=list)

    def __str__(self) -> str:
        head = "PASS" if self.ok else "FAIL"
        lines = [f"audit {head}: {self.edges_claimed} unit distances among "
                 f"{self.distinct_points}/{self.n} distinct points, "
                 f"min separation {self.min_separation:.4g}, "
                 f"max edge error {self.max_edge_error:.1e}"]
        lines += [f"  - {p}" for p in self.problems]
        return "\n".join(lines)


def audit(X: np.ndarray, edges: np.ndarray | None = None, epsilon: float = 1e-9,
          min_sep_required: float = 0.0, dup_tol: float = DUP_TOL) -> Audit:
    """Check that a configuration's unit-distance count is not inflated.

    Verifies, independently of whatever produced the configuration:

    * no two balls are within ``dup_tol`` -- overlapping points would let a
      single location be counted many times over;
    * recounting from the coordinates alone reproduces the claimed edges;
    * recounting *after* merging any coincident balls gives the same answer,
      so the count does not depend on duplicates;
    * every claimed edge really is within ``epsilon`` of unit length, and
      each is listed once as an ordered pair.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    problems: list[str] = []

    reps, _ = merge_duplicates(X, dup_tol)
    sep = min_separation(X)
    if reps.shape[0] < n:
        problems.append(f"{n - reps.shape[0]} ball(s) coincide within {dup_tol:g}; "
                        f"the count is inflated by duplicate points")
    if min_sep_required > 0.0 and sep < min_sep_required:
        problems.append(f"min separation {sep:.4g} is below the required "
                        f"{min_sep_required:.4g}")

    recount = count_unit_distances(X, epsilon)
    distinct_count = count_unit_distances(reps, epsilon)
    if distinct_count != recount:
        problems.append(f"count drops from {recount} to {distinct_count} once "
                        f"coincident balls are merged")

    if edges is None:
        edges = unit_edges(X, epsilon)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    claimed = int(edges.shape[0])

    lengths = edge_lengths(X, edges)
    err = float(np.abs(lengths - 1.0).max()) if lengths.size else 0.0
    if lengths.size and err > epsilon:
        problems.append(f"{int((np.abs(lengths - 1.0) > epsilon).sum())} claimed "
                        f"edge(s) are not within {epsilon:g} of unit length")
    if edges.shape[0] and not (edges[:, 0] < edges[:, 1]).all():
        problems.append("edge list is not made of ordered pairs i < j")
    if len({tuple(e) for e in edges.tolist()}) != claimed:
        problems.append("edge list contains duplicates")
    if claimed != recount:
        problems.append(f"claimed {claimed} edges but recounting the "
                        f"coordinates gives {recount}")

    return Audit(ok=not problems, n=n, distinct_points=int(reps.shape[0]),
                 min_separation=sep, edges_claimed=claimed,
                 edges_recounted=recount,
                 edges_among_distinct_points=distinct_count,
                 max_edge_error=err, problems=problems)
