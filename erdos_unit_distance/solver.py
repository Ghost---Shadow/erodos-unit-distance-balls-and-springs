"""End-to-end pipeline: anneal, extract the graph, then snap it exact.

The relaxation is good at *combinatorics* and bad at *precision*: it will
happily park a spring at 1.0004.  So the run is split in two.

1. **Anneal** the complete spring graph (:mod:`sim`) to find which pairs
   want to be unit.
2. **Project**: freeze that edge set and solve the pure geometry problem
   exactly (:mod:`polish`), then look for near-misses that the tightened
   geometry brought into range and try growing the edge set.

Step 2 is what turns "about 1" into "1 to fourteen decimal places", and
the growth loop typically buys a few extra edges for free.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from . import init as init_mod
from .counting import (audit, count_unit_distances, degree_sequence,
                       min_separation, unit_edges)
from .energy import EnergyParams
from .known import default_r_min
from .polish import polish_and_prune
from .sim import Schedule, Simulation

FINAL_TOL = 1e-9


@dataclass
class SolveResult:
    n: int
    edges: int
    X: np.ndarray = field(repr=False)
    edge_list: np.ndarray = field(repr=False)
    max_edge_error: float = 0.0
    min_separation: float = 0.0
    max_degree: int = 0
    edges_before_polish: int = 0
    seed: int | None = None
    trial: int = 0
    valid: bool = True
    """False if the audit found anything wrong -- most importantly balls
    close enough to count as the same point, which would inflate the
    count.  Always check this before believing ``edges``."""
    distinct_points: int = 0
    """How many of the n balls are actually distinct locations."""
    problems: list = field(default_factory=list)
    """Human-readable reasons the audit failed, empty when it passed."""

    def to_json(self) -> dict:
        d = asdict(self)
        d["X"] = self.X.tolist()
        d["edge_list"] = self.edge_list.tolist()
        return d

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_json(), fh, indent=2)


def project(X: np.ndarray, extract_tol: float = 0.02, grow_tol: float = 0.02,
            r_min: float = 0.12, rounds: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Freeze the near-unit springs, solve them exactly, then try to grow.

    After the exact solve the geometry has tightened, which usually pulls
    a few more springs into range; those get added and the solve repeated
    for as long as it keeps paying off.

    Returns ``(X_exact, edges)``.
    """
    X = np.asarray(X, dtype=np.float64)
    edges = unit_edges(X, extract_tol)
    X, edges, _ = polish_and_prune(X, edges, r_min=r_min)
    best_count = count_unit_distances(X, FINAL_TOL)

    for _ in range(rounds):
        cand = unit_edges(X, grow_tol)
        if cand.shape[0] <= edges.shape[0]:
            break
        Xn, en, _ = polish_and_prune(X, cand, r_min=r_min)
        cn = count_unit_distances(Xn, FINAL_TOL)
        if cn <= best_count or min_separation(Xn) < 0.5 * r_min:
            break
        X, edges, best_count = Xn, en, cn

    return X, unit_edges(X, FINAL_TOL)


def solve_once(n: int, seed: int | None = None, init: str = "random",
               params: EnergyParams | None = None, schedule: Schedule | None = None,
               integrator: str = "md", epsilon: float = 1e-3, jitter: float = 0.0,
               extract_tol: float = 0.02, cycles: int = 3, reheat: float = 0.45,
               rescue_every: int = 100, callback=None, trial: int = 0) -> SolveResult:
    """One full run: anneal, project, then reheat and repeat.

    ``cycles`` basin-hopping rounds are run.  Each round anneals, projects
    onto exact unit distances, then restarts the anneal from that exact
    configuration with ``sigma0`` scaled by ``reheat``.  Starting a round
    from a state that already lies on the unit-distance manifold is worth
    far more than annealing longer: the sweep showed extra steps stop
    paying off around 2500, because what limits a run is which basin it
    fell into, not how carefully it descended.
    """
    rng = np.random.default_rng(seed)
    params = params or EnergyParams(r_min=default_r_min(n))
    schedule = schedule or Schedule()
    X = init_mod.make(init, n, rng, jitter=jitter)

    best_X, best_edges, before = X, np.zeros((0, 2), dtype=np.int64), 0
    for cycle in range(max(cycles, 1)):
        sched = schedule if cycle == 0 else replace(
            schedule, sigma0=max(schedule.sigma0 * reheat, schedule.sigma1 * 2.0))
        sim = Simulation(X, params=params, schedule=sched, integrator=integrator,
                         epsilon=epsilon, rng=rng, rescue_every=rescue_every)
        sim.run(callback=callback)
        before = max(before, count_unit_distances(sim.best_X, epsilon))

        # The best-scoring intermediate state and the final state can
        # project differently; keep whichever yields more exact edges.
        for candidate in (sim.best_X, sim.X):
            Xp, ep = project(candidate, extract_tol=extract_tol, r_min=params.r_min)
            if ep.shape[0] > best_edges.shape[0] and min_separation(Xp) >= 0.5 * params.r_min:
                best_X, best_edges = Xp, ep

        # Reheat from the exact configuration, nudged off the fixed point.
        X = best_X + rng.normal(scale=0.05, size=best_X.shape)

    X, edges = best_X, best_edges
    # Re-derive everything from the coordinates alone, so a bug in the
    # pipeline cannot quietly inflate the reported number.
    checked = audit(X, edges, epsilon=FINAL_TOL,
                    min_sep_required=0.5 * params.r_min)
    err = checked.max_edge_error
    sep = checked.min_separation
    return SolveResult(
        n=n,
        edges=int(edges.shape[0]),
        X=X,
        edge_list=edges,
        max_edge_error=err,
        min_separation=sep,
        max_degree=int(degree_sequence(n, edges).max()) if n else 0,
        edges_before_polish=before,
        seed=seed,
        trial=trial,
        valid=checked.ok,
        distinct_points=checked.distinct_points,
        problems=list(checked.problems),
    )


def solve_multi(n: int, trials: int = 8, seed: int = 0, progress=None, **kw) -> SolveResult:
    """Run several independent restarts and keep the best."""
    best: SolveResult | None = None
    for t in range(trials):
        res = solve_once(n, seed=seed + t, trial=t, **kw)
        # A configuration with near-duplicate balls always loses to a
        # valid one, however many "unit distances" it claims.
        if best is None or (res.valid, res.edges) > (best.valid, best.edges):
            best = res
        if progress is not None:
            progress(t, res, best)
    assert best is not None
    return best
