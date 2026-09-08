"""The annealed ball-and-spring relaxation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .counting import count_unit_distances, min_separation
from .energy import EnergyParams, energy_and_grad, soft_count

INTEGRATORS = ("md", "adam", "gd")


@dataclass
class Schedule:
    """Annealing schedule.

    ``sigma`` is the strain scale over which a spring stays stiff.  It
    starts wide -- every pair in the complete graph pulls, so the assembly
    behaves like one big frustrated Hooke network -- and narrows
    geometrically until only springs already close to unit exert force.
    The temperature adds Langevin kicks so the network can escape shallow
    local minima early on, and is cooled to zero.

    The defaults were tuned by sweeping against the n <= 14 configurations
    whose true optimum is known.  ``sigma0`` matters most and wants to be
    *small*: starting at 0.5 lets every pair in the complete graph pull at
    once and the assembly crushes itself into a blob, scoring well under
    half of what 0.18 reaches.  Extra steps buy almost nothing beyond
    ~2500 -- the binding constraint is which basin the run lands in, which
    is what the restarts and reheat cycles are for.
    """

    steps: int = 2500
    sigma0: float = 0.18
    sigma1: float = 0.015
    temp0: float = 0.06
    temp1: float = 0.0
    dt: float = 0.07
    damping: float = 1.5
    lr: float = 0.05

    def _t(self, step: int) -> float:
        return min(max(step / max(self.steps - 1, 1), 0.0), 1.0)

    def sigma(self, step: int) -> float:
        return float(self.sigma0 * (self.sigma1 / self.sigma0) ** self._t(step))

    def temperature(self, step: int) -> float:
        t = self._t(step)
        return float(self.temp0 * (1.0 - t) + self.temp1 * t)

    # -- self-similar scalings ---------------------------------------- #
    #
    # A gaussian well of fixed depth and width sigma has small-strain
    # stiffness depth/sigma**2, so its oscillation period scales like
    # sigma.  Scaling dt with sigma and damping with 1/sigma keeps both
    # dt*omega and dt*gamma constant: the dynamics look identical at
    # every stage of the anneal, just at a finer geometric scale.  Without
    # this the integrator goes unstable as soon as the well narrows.

    def dt_eff(self, step: int) -> float:
        return float(self.dt * self.sigma(step) / self.sigma0)

    def damping_eff(self, step: int) -> float:
        return float(self.damping * self.sigma0 / self.sigma(step))

    def lr_eff(self, step: int) -> float:
        """Adam moves ~lr per coordinate per step regardless of gradient
        scale, so the learning rate carries the sigma scaling directly."""
        return float(self.lr * self.sigma(step) / self.sigma0)


@dataclass
class Stats:
    step: int = 0
    sigma: float = 0.0
    temperature: float = 0.0
    energy: float = 0.0
    grad_norm: float = 0.0
    soft_edges: float = 0.0
    edges: int = 0
    best_edges: int = 0
    min_sep: float = 0.0


class Simulation:
    """A complete graph of unit-rest-length springs, relaxed under annealing.

    One :meth:`step` advances the assembly by a single integrator step.
    The object is designed to be driven either by a render loop (a few
    steps per frame) or by :meth:`run` in headless mode.
    """

    def __init__(self, X0: np.ndarray, params: EnergyParams | None = None,
                 schedule: Schedule | None = None, integrator: str = "md",
                 epsilon: float = 1e-3, rng: np.random.Generator | None = None,
                 eval_every: int = 20, rescue_every: int = 100):
        if integrator not in INTEGRATORS:
            raise ValueError(f"unknown integrator {integrator!r}; expected one of {INTEGRATORS}")
        self.X0 = np.array(X0, dtype=np.float64, copy=True)
        self.X = np.array(X0, dtype=np.float64, copy=True)
        self.params = params or EnergyParams()
        self.schedule = schedule or Schedule()
        self.integrator = integrator
        self.epsilon = epsilon
        self.rng = rng or np.random.default_rng()
        self.eval_every = eval_every
        self.rescue_every = rescue_every
        self.rescued = 0

        self.n = self.X.shape[0]
        self.V = np.zeros_like(self.X)
        self._m = np.zeros_like(self.X)          # Adam first moment
        self._v = np.zeros_like(self.X)          # Adam second moment
        self.step_index = 0
        self.best_X = self.X.copy()
        self.best_edges = count_unit_distances(self.X, epsilon)
        self.stats = Stats(best_edges=self.best_edges)

        p = self.params.with_sigma(self.schedule.sigma(0))
        self.energy, self.grad = energy_and_grad(self.X, p)

    # ------------------------------------------------------------------ #

    @property
    def sigma(self) -> float:
        return self.schedule.sigma(self.step_index)

    @property
    def progress(self) -> float:
        return min(self.step_index / max(self.schedule.steps, 1), 1.0)

    @property
    def done(self) -> bool:
        return self.step_index >= self.schedule.steps

    def reset(self, X0: np.ndarray | None = None) -> None:
        """Restart the run, optionally from a fresh configuration."""
        if X0 is not None:
            self.X0 = np.array(X0, dtype=np.float64, copy=True)
        self.X = self.X0.copy()
        self.V[:] = 0.0
        self._m[:] = 0.0
        self._v[:] = 0.0
        self.step_index = 0
        self.best_X = self.X.copy()
        self.best_edges = count_unit_distances(self.X, self.epsilon)
        self.rescued = 0
        self.energy, self.grad = energy_and_grad(self.X, self._params_now())

    def _params_now(self) -> EnergyParams:
        return self.params.with_sigma(self.sigma)

    def rescue_detached(self) -> int:
        """Move balls that have snapped every spring back into the assembly.

        A ball with no other ball within reach feels no force at all: it
        is inert, drifts on whatever velocity it had, and contributes
        nothing.  The weak centroid spring is far too soft to reel it back
        within a run, so it is simply put back to work at a random spot
        inside the assembly.

        Returns how many balls were moved.
        """
        # A ball is inert once nothing is close enough to pull on it.  With
        # spring failure on, that is exactly break_hi.  With it off, the
        # saturating laws still die away past a few sigma of strain.
        #
        # The reach is deliberately pinned to sigma0, not the current
        # sigma.  Letting it shrink with the anneal makes it approach 1,
        # at which point the rescue starts relocating balls that are still
        # doing useful work -- which destroys structure and costs real
        # results (it dropped n=7 from its optimum of 12 to 11).
        reach = min(self.params.break_hi, 1.0 + 4.0 * self.schedule.sigma0)
        if not np.isfinite(reach) or self.n < 2:
            return 0

        X = self.X
        nearest = np.full(self.n, np.inf)
        for lo in range(0, self.n, 512):
            hi = min(lo + 512, self.n)
            diff = X[lo:hi, None, :] - X[None, :, :]
            d = np.sqrt(np.einsum("bnk,bnk->bn", diff, diff))
            d[np.arange(hi - lo), np.arange(lo, hi)] = np.inf
            nearest[lo:hi] = d.min(axis=1)

        lost = np.nonzero(nearest > reach)[0]
        if lost.size == 0:
            return 0

        keep = np.setdiff1d(np.arange(self.n), lost)
        centre = X[keep].mean(axis=0) if keep.size else X.mean(axis=0)
        radius = np.sqrt(max(keep.size, 1) / (np.pi * 1.1547))
        theta = self.rng.uniform(0.0, 2.0 * np.pi, lost.size)
        r = radius * np.sqrt(self.rng.random(lost.size))
        self.X[lost] = centre + np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
        self.V[lost] = 0.0
        self.rescued += int(lost.size)
        self.energy, self.grad = energy_and_grad(self.X, self._params_now())
        return int(lost.size)

    # ------------------------------------------------------------------ #

    def step(self) -> Stats:
        """Advance one integrator step and return the current statistics."""
        sch = self.schedule
        p = self._params_now()
        temp = sch.temperature(self.step_index)

        if self.integrator == "md":
            # Velocity Verlet with viscous damping, plus Langevin kicks.
            # Force is minus the gradient; unit ball masses.
            dt = sch.dt_eff(self.step_index)
            gamma = sch.damping_eff(self.step_index)
            self.V += 0.5 * dt * (-self.grad)
            self.X += dt * self.V
            self.energy, self.grad = energy_and_grad(self.X, p)
            self.V += 0.5 * dt * (-self.grad)
            self.V *= np.exp(-gamma * dt)
            if temp > 0.0:
                self.V += np.sqrt(2.0 * gamma * temp * dt) * self.rng.normal(size=self.V.shape)
            # Drop net drift so the assembly stays put on screen.
            self.V -= self.V.mean(axis=0, keepdims=True)

        elif self.integrator == "adam":
            lr = sch.lr_eff(self.step_index)
            self._m = 0.9 * self._m + 0.1 * self.grad
            self._v = 0.999 * self._v + 0.001 * self.grad ** 2
            t = self.step_index + 1
            mhat = self._m / (1.0 - 0.9 ** t)
            vhat = self._v / (1.0 - 0.999 ** t)
            self.X -= lr * mhat / (np.sqrt(vhat) + 1e-12)
            if temp > 0.0:
                self.X += np.sqrt(temp) * lr * self.rng.normal(size=self.X.shape)
            self.energy, self.grad = energy_and_grad(self.X, p)

        else:  # plain gradient descent
            # Gradients scale like depth/sigma, so the step scales like sigma^2.
            lr = sch.lr * (self.sigma / sch.sigma0) ** 2
            self.X -= lr * self.grad
            if temp > 0.0:
                self.X += np.sqrt(temp) * lr * self.rng.normal(size=self.X.shape)
            self.energy, self.grad = energy_and_grad(self.X, p)

        self.step_index += 1
        if self.rescue_every and self.step_index % self.rescue_every == 0:
            self.rescue_detached()

        s = self.stats
        s.step = self.step_index
        s.sigma = self.sigma
        s.temperature = temp
        s.energy = self.energy
        s.grad_norm = float(np.linalg.norm(self.grad))

        if self.step_index % self.eval_every == 0 or self.done:
            s.edges = count_unit_distances(self.X, self.epsilon)
            s.soft_edges = soft_count(self.X, max(self.sigma, 1e-3))
            s.min_sep = min_separation(self.X)
            if s.edges > self.best_edges:
                self.best_edges = s.edges
                self.best_X = self.X.copy()
        s.best_edges = self.best_edges
        return s

    def run(self, callback=None, every: int = 200) -> Stats:
        """Run the whole schedule headlessly."""
        while not self.done:
            s = self.step()
            if callback is not None and self.step_index % every == 0:
                callback(s)
        return self.stats
