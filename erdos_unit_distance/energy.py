"""Total energy and gradient of the complete-graph spring assembly.

    E(X) = sum_{i<j} spring(d_ij - 1)      complete graph, rest length 1
         + sum_{i<j} core(d_ij)            short-range repulsion
         + confinement(X)                  weak pull toward the centroid

Springs outside ``[break_lo, break_hi]`` have snapped and contribute no
force at all.  The core repulsion is deliberately *not* subject to
failure: it is what stops balls that have snapped every spring between
them from simply passing through each other.

The pair sums are dense (O(n^2)) but evaluated in row blocks so peak
memory stays bounded, and the gradient is assembled with the identity

    grad_i = (sum_j C_ij) * x_i  -  sum_j C_ij * x_j

with ``C_ij = (dE/dd)_ij / d_ij``, i.e. one matrix-vector product instead
of materialising the n x n x 2 difference tensor.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .springs import apply_failure, soft_edge_weight, spring_energy_force

_TINY = 1e-12


@dataclass
class EnergyParams:
    """Physical constants of the assembly."""

    law: str = "gaussian"
    depth: float = 1.0
    """Energy cost of a fully violated spring = reward for satisfying one.
    For the ``hooke`` law this is the stiffness instead."""
    sigma: float = 0.30
    """Strain scale over which springs stay stiff.  Annealed during a run."""
    r_min: float = 0.12
    """Balls repel each other below this separation."""
    k_core: float = 1.0
    """Core stiffness, as a multiple of the current spring stiffness.

    It has to be expressed relatively: a saturating spring's stiffness is
    ``depth / sigma**2``, which grows by orders of magnitude during the
    anneal.  A fixed core stiffness gets overwhelmed and the assembly
    collapses into a few coincident clusters sitting a unit apart -- a
    configuration that scores a spurious floor(n^2/3)-ish 'unit distances'
    purely by stacking duplicate points."""
    k_conf: float = 2e-3
    """Weak spring to the centroid; keeps detached balls from drifting off."""
    break_lo: float = 0.0
    """A spring squeezed below this length snaps and exerts no force.
    0 disables the lower threshold."""
    break_hi: float = float("inf")
    """A spring stretched beyond this length snaps and exerts no force.
    Infinite disables the upper threshold.

    Failure is off by default: the saturating spring laws already let a
    badly violated spring fade out smoothly, so the hard cutoff is a
    second, blunter mechanism on top.  Turn it on with, say,
    ``break_lo=0.5, break_hi=2.0``."""
    block: int = 512
    """Row-block size for the pair loops."""

    def with_sigma(self, sigma: float) -> "EnergyParams":
        return replace(self, sigma=sigma)

    def stiffness_scale(self) -> float:
        """Small-strain stiffness of a single spring under the current law."""
        if self.law == "hooke":
            return self.depth
        return self.depth / (self.sigma * self.sigma)


def energy_and_grad(X: np.ndarray, p: EnergyParams):
    """Return ``(energy, gradient)`` for ball positions ``X`` of shape (n, 2)."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    grad = np.zeros_like(X)
    energy = 0.0

    for lo in range(0, n, p.block):
        hi = min(lo + p.block, n)
        blk = X[lo:hi]                                  # (b, 2)
        diff = blk[:, None, :] - X[None, :, :]          # (b, n, 2)
        d = np.sqrt(np.einsum("bnk,bnk->bn", diff, diff))
        # Park self pairs at the rest length so they contribute no energy
        # and no force, instead of feeding inf/nan through the spring law.
        self_pair = (np.arange(hi - lo), np.arange(lo, hi))
        d[self_pair] = 1.0

        e_spring, f_spring = spring_energy_force(d - 1.0, p.law, p.depth, p.sigma)
        # Springs pushed too far past their rest length give up entirely.
        e_spring, f_spring = apply_failure(d - 1.0, e_spring, f_spring, p.law,
                                           p.depth, p.sigma, p.break_lo, p.break_hi)

        # Short-range core: keeps distinct balls from sitting on top of
        # each other.  Scaled to the spring stiffness so it cannot be
        # overwhelmed as the well narrows.
        k_core = p.k_core * p.stiffness_scale()
        gap = p.r_min - d
        overlap = gap > 0.0
        e_core = np.where(overlap, k_core * gap * gap, 0.0)
        f_core = np.where(overlap, -2.0 * k_core * gap, 0.0)

        e_pair = e_spring + e_core
        f_pair = f_spring + f_core
        e_pair[self_pair] = 0.0
        f_pair[self_pair] = 0.0

        energy += 0.5 * float(e_pair.sum())             # each pair seen twice
        C = f_pair / np.maximum(d, _TINY)
        grad[lo:hi] = C.sum(axis=1)[:, None] * blk - C @ X

    # Confinement about the centroid (translation invariant).
    centred = X - X.mean(axis=0, keepdims=True)
    energy += 0.5 * p.k_conf * float((centred * centred).sum())
    grad += p.k_conf * centred

    return energy, grad


def soft_count(X: np.ndarray, sigma: float, block: int = 512) -> float:
    """Differentiable surrogate for the number of unit distances.

    Sums a Gaussian bump of width ``sigma`` over every pair's strain, so a
    spring exactly at unit length contributes 1 and a badly violated one
    contributes ~0.  Equals the hard count in the ``sigma -> 0`` limit.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    total = 0.0
    for lo in range(0, n, block):
        hi = min(lo + block, n)
        diff = X[lo:hi, None, :] - X[None, :, :]
        d = np.sqrt(np.einsum("bnk,bnk->bn", diff, diff))
        w = soft_edge_weight(d - 1.0, sigma)
        w[np.arange(hi - lo), np.arange(lo, hi)] = 0.0
        total += float(w.sum())
    return 0.5 * total


def pair_distances(X: np.ndarray) -> np.ndarray:
    """Condensed vector of all n(n-1)/2 pair distances (upper triangle order)."""
    X = np.asarray(X, dtype=np.float64)
    diff = X[:, None, :] - X[None, :, :]
    d = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    iu = np.triu_indices(X.shape[0], k=1)
    return d[iu]
