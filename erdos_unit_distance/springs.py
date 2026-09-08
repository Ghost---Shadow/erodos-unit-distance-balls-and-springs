"""Spring laws for the complete-graph ball-and-spring model.

Every one of the n(n-1)/2 pairs carries a spring with **rest length 1**.
A spring's state is its signed strain

    s = d - 1

and a spring is called an *edge* when ``|s| <= epsilon``.

A spring law maps strain to (energy, dE/dd).  What separates the laws is
the *tail*: on a complete graph a pure Hooke spring at d = 5 pulls five
times harder than one at d = 1.2, so far-off pairs dominate the gradient
and crush everything into a blob.  The saturating laws let a badly
violated spring soften and effectively detach, which is what lets a
frustrated network settle into a state where *many* springs are
individually satisfied.

Amplitude convention
--------------------
The saturating laws are normalised so a spring costs ``depth`` when it is
hopelessly violated and 0 when it sits exactly at unit length::

    E(0) = 0        E(|s| >> sigma) -> depth

So ``depth`` is the reward for satisfying one spring, and the total
energy is ``depth * (#pairs - #satisfied)``: minimising it *is*
maximising the (softened) unit-distance count.  Critically the amplitude
does **not** depend on ``sigma``.  Holding the small-strain stiffness
fixed instead would make the amplitude vanish as ``sigma`` anneals to
zero, and the objective would evaporate exactly when it matters most.

The price is that the effective small-strain stiffness ``depth / sigma**2``
*grows* as the well narrows.  The MD integrator absorbs that by scaling
its timestep with ``sigma`` (see :class:`~erdos_unit_distance.sim.Schedule`);
Adam is invariant to it for free.
"""

from __future__ import annotations

import numpy as np

LAWS = ("gaussian", "hooke", "gated", "lorentz")


def spring_energy_force(s: np.ndarray, law: str, depth: float, sigma: float):
    """Return ``(energy, dE_dd)`` elementwise for strains ``s = d - 1``.

    Parameters
    ----------
    s : array
        Signed strain of every pair.
    law : {'gaussian', 'hooke', 'gated', 'lorentz'}
    depth : float
        Energy cost of a fully violated spring, i.e. the reward for
        satisfying one.  For ``hooke`` it is the stiffness instead.
    sigma : float
        Strain scale over which a spring stays stiff.  Annealing it from
        large to small interpolates from a true complete Hooke graph down
        to a network where only near-unit springs exert any force.
        Ignored by ``hooke``.
    """
    if law == "hooke":
        # Textbook spring, never detaches.  Energy unbounded in strain.
        return 0.5 * depth * s * s, depth * s

    if law == "gaussian":
        # Well of fixed depth and shrinking width.  Force peaks at |s| = sigma
        # and dies exponentially beyond, so hopeless pairs stop pulling.
        w = np.exp(-0.5 * (s / sigma) ** 2)
        return depth * (1.0 - w), depth * s * w / (sigma * sigma)

    if law == "lorentz":
        # Same fixed depth, but an algebraic tail: force decays like 1/s^3
        # rather than exponentially, so distant balls still feel a faint
        # pull toward the unit shell and are less likely to be stranded.
        s2 = s * s
        q = sigma * sigma + s2
        return depth * s2 / q, 2.0 * depth * s * sigma * sigma / (q * q)

    if law == "gated":
        # Literal reading of "only springs within epsilon are edges": a
        # spring outside the gate is simply switched off.  Matches the
        # gaussian's small-strain stiffness.  The energy is discontinuous
        # at the gate -- fine for a damped relaxation, but not a
        # differentiable objective.
        k = depth / (sigma * sigma)
        inside = np.abs(s) <= sigma
        return np.where(inside, 0.5 * k * s * s, 0.0), np.where(inside, k * s, 0.0)

    raise ValueError(f"unknown spring law {law!r}; expected one of {LAWS}")


def soft_edge_weight(s: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth membership in [0, 1] of each spring in the 'is an edge' set.

    Summed over all pairs this is a differentiable surrogate for the
    unit-distance count, and it agrees with the hard epsilon-threshold
    count in the ``sigma -> 0`` limit.
    """
    return np.exp(-0.5 * (s / sigma) ** 2)


def apply_failure(s: np.ndarray, e: np.ndarray, f: np.ndarray, law: str,
                  depth: float, sigma: float, break_lo: float, break_hi: float):
    """Snap springs whose length has left ``[break_lo, break_hi]``.

    A failed spring exerts **zero force**.  Its energy is held at the
    value it had the instant it broke, so the energy stays continuous
    across the threshold and simply plateaus -- there is no force step and
    no spurious well at the break point that balls could get caught in.

    This is a blunter version of what the saturating laws do smoothly, and
    it composes with them: ``gaussian`` already ignores anything far from
    unit, whereas ``hooke`` does not, so pairing ``hooke`` with a failure
    range is what makes a literal complete Hooke graph behave at all.

    Distances, not strains, define the range: ``break_lo = 0.5`` and
    ``break_hi = 2.0`` mean a spring gives up once it is squeezed to half
    its rest length or stretched to double it.
    """
    if break_lo <= 0.0 and not np.isfinite(break_hi):
        return e, f

    e = e.copy()
    f = f.copy()
    d = s + 1.0

    if break_lo > 0.0:
        snapped = d < break_lo
        if snapped.any():
            e_lo = spring_energy_force(np.array([break_lo - 1.0]), law, depth, sigma)[0][0]
            e[snapped] = e_lo
            f[snapped] = 0.0

    if np.isfinite(break_hi):
        snapped = d > break_hi
        if snapped.any():
            e_hi = spring_energy_force(np.array([break_hi - 1.0]), law, depth, sigma)[0][0]
            e[snapped] = e_hi
            f[snapped] = 0.0

    return e, f
