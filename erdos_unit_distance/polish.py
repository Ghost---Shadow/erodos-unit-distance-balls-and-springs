"""Snap a nearly-unit-distance configuration onto exact unit distances.

The annealed relaxation finds the *combinatorics* -- which pairs want to
be at distance 1 -- but leaves each edge off by ~1e-3.  This module takes
that edge set as fixed and solves the geometry exactly:

    minimise  sum over edges     (dist - 1)^2
            + w^2 * sum over close pairs  (r_min - dist)_+^2

with Levenberg-Marquardt, which converges quadratically and routinely
drives the edge residuals to machine precision.

The second sum is not decoration.  Without it LM cheerfully drives points
into *exact coincidence*: three coincident clusters a unit apart satisfy
all of their mutual "unit distance" equations and score ~n^2/3, which is
a counting artefact rather than a point set -- the Erdos problem is about
n *distinct* points.  The barrier keeps the solution honest.

The three-dimensional gauge freedom (translation + rotation) leaves the
normal equations singular; LM damping handles that with no explicit
pinning.  Only numpy is required.
"""

from __future__ import annotations

import numpy as np

_TINY = 1e-12
_BLOCK = 512
#: Above this many balls, solve the LM step matrix-free with CG instead of
#: forming and factorising the dense (2n x 2n) normal equations.
DENSE_LIMIT = 250


def residuals(X: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Edge length minus 1, for each edge."""
    if edges.shape[0] == 0:
        return np.zeros(0)
    v = X[edges[:, 0]] - X[edges[:, 1]]
    return np.sqrt((v * v).sum(axis=1)) - 1.0


def close_pairs(X: np.ndarray, r_min: float) -> np.ndarray:
    """Pairs i < j currently closer than ``r_min``."""
    n = X.shape[0]
    rows, cols = [], []
    for lo in range(0, n, _BLOCK):
        hi = min(lo + _BLOCK, n)
        diff = X[lo:hi, None, :] - X[None, :, :]
        d = np.sqrt(np.einsum("bnk,bnk->bn", diff, diff))
        hit = (d < r_min) & (np.arange(lo, hi)[:, None] < np.arange(n)[None, :])
        r, c = np.nonzero(hit)
        rows.append(r + lo)
        cols.append(c)
    if not rows:
        return np.zeros((0, 2), dtype=np.int64)
    return np.stack([np.concatenate(rows), np.concatenate(cols)], axis=1)


def _unit_vectors(X: np.ndarray, pairs: np.ndarray):
    """Return ``(distance, unit_vector)`` for each listed pair."""
    if pairs.shape[0] == 0:
        return np.zeros(0), np.zeros((0, 2))
    v = X[pairs[:, 0]] - X[pairs[:, 1]]
    d = np.sqrt((v * v).sum(axis=1))
    return d, v / np.maximum(d, _TINY)[:, None]


class _Problem:
    """Residuals, Jacobian products and cost for one fixed active set.

    Rows are the edge constraints followed by the core-barrier
    constraints.  The barrier's active set is refreshed between outer LM
    iterations, never inside the trust-region search, so the costs being
    compared always refer to the same function.
    """

    def __init__(self, edges, core, r_min, w_core):
        self.r_min = r_min
        self.w_core = w_core
        self.n_edge = edges.shape[0]
        self.pairs = np.vstack([edges, core]) if core.shape[0] else edges

    def residual(self, X):
        d, _ = _unit_vectors(X, self.pairs)
        r = d - 1.0
        if self.pairs.shape[0] > self.n_edge:
            r[self.n_edge:] = self.w_core * (self.r_min - d[self.n_edge:])
        return r

    def cost(self, X):
        r = self.residual(X)
        return float(r @ r)

    def _row_grads(self, X):
        """d(residual)/d(distance) times the pair unit vector, per row."""
        _, u = _unit_vectors(X, self.pairs)
        sign = np.ones(self.pairs.shape[0])
        sign[self.n_edge:] = -self.w_core
        return u * sign[:, None]

    def normal_equations(self, X, r, n):
        """Dense J^T J and J^T r, for small problems."""
        m = self.pairs.shape[0]
        J = np.zeros((m, 2 * n))
        g = self._row_grads(X)
        rows = np.arange(m)
        for k in range(2):
            np.add.at(J, (rows, 2 * self.pairs[:, 0] + k), g[:, k])
            np.add.at(J, (rows, 2 * self.pairs[:, 1] + k), -g[:, k])
        return J.T @ J, J.T @ r

    def operators(self, X, n):
        """Matrix-free ``J @ v`` and ``J.T @ y``, for large problems."""
        g = self._row_grads(X)
        i, j = self.pairs[:, 0], self.pairs[:, 1]

        def jv(v):
            V = v.reshape(n, 2)
            return np.einsum("mk,mk->m", g, V[i] - V[j])

        def jty(y):
            out = np.zeros((n, 2))
            contrib = g * y[:, None]
            np.add.at(out, i, contrib)
            np.add.at(out, j, -contrib)
            return out.ravel()

        return jv, jty


def _cg_step(jv, jty, rhs, lam, n, iters=200, tol=1e-10):
    """Conjugate gradients on ``(J^T J + lam I) x = rhs``, matrix-free."""
    x = np.zeros(2 * n)
    r = rhs.copy()
    p = r.copy()
    rs = float(r @ r)
    if rs == 0.0:
        return x
    target = tol * tol * rs
    for _ in range(iters):
        Ap = jty(jv(p)) + lam * p
        denom = float(p @ Ap)
        if denom <= 0.0:
            break
        alpha = rs / denom
        x += alpha * p
        r -= alpha * Ap
        rs_new = float(r @ r)
        if rs_new < target:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x


def polish(X: np.ndarray, edges: np.ndarray, r_min: float = 0.12,
           w_core: float = 4.0, iters: int = 60, tol: float = 1e-24):
    """Levenberg-Marquardt refinement holding ``edges`` at length 1.

    ``r_min`` is the hard core that keeps balls distinct; pass 0 to
    disable it.  Returns ``(X_refined, max_abs_edge_residual)``.
    ``X`` is not modified.
    """
    X = np.array(X, dtype=np.float64, copy=True)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if edges.shape[0] == 0:
        return X, 0.0

    n = X.shape[0]
    dense = n <= DENSE_LIMIT
    lam = 1e-6

    for _ in range(iters):
        core = close_pairs(X, r_min) if r_min > 0.0 else np.zeros((0, 2), dtype=np.int64)
        prob = _Problem(edges, core, r_min, w_core)
        r = prob.residual(X)
        cost = float(r @ r)
        if cost <= tol:
            break

        if dense:
            JTJ, JTr = prob.normal_equations(X, r, n)
        else:
            jv, jty = prob.operators(X, n)
            JTr = jty(r)

        improved = False
        for _ in range(30):                     # trust-region search on lam
            if dense:
                try:
                    step = np.linalg.solve(JTJ + lam * np.eye(2 * n), -JTr)
                except np.linalg.LinAlgError:
                    lam *= 10.0
                    continue
            else:
                step = _cg_step(jv, jty, -JTr, lam, n)

            cand = X + step.reshape(-1, 2)
            if prob.cost(cand) < cost:
                X = cand
                lam = max(lam * 0.3, 1e-12)
                improved = True
                break
            lam *= 10.0

        if not improved:
            break                               # no downhill step exists

    return X, float(np.abs(residuals(X, edges)).max())


def polish_and_prune(X: np.ndarray, edges: np.ndarray, r_min: float = 0.12,
                     keep_tol: float = 1e-9, rounds: int = 5, iters: int = 60):
    """Polish, drop edges that refuse to converge, and repeat.

    A proposed edge set can be geometrically unrealisable: the relaxation
    may have suggested pairs that cannot all sit at unit length at once
    while the balls stay distinct.  Dropping the worst offenders and
    re-solving yields a set that *is* realisable.

    Returns ``(X_refined, surviving_edges, max_abs_residual)``.
    """
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    for _ in range(rounds):
        X, err = polish(X, edges, r_min=r_min, iters=iters)
        if err <= keep_tol or edges.shape[0] == 0:
            break
        r = np.abs(residuals(X, edges))
        keep = r <= max(keep_tol, 0.5 * r.max())
        if keep.all():
            break
        edges = edges[keep]
    err = float(np.abs(residuals(X, edges)).max()) if edges.shape[0] else 0.0
    return X, edges, err
