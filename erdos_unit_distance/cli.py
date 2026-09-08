"""Command line entry point.

    python run.py --n 24                    live 2D view
    python run.py --n 24 --headless         no window, multi-restart search
    python run.py --benchmark               score against the known optima
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from . import init as init_mod
from .energy import EnergyParams
from .known import default_r_min, known_optimum, summarise
from .sim import INTEGRATORS, Schedule, Simulation
from .solver import solve_multi
from .springs import LAWS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="erdos-unit-distance",
        description="Maximise unit distances among n points with a complete "
                    "graph of rest-length-1 springs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("problem")
    g.add_argument("--n", type=int, default=20, help="number of balls")
    g.add_argument("--seed", type=int, default=0, help="random seed")
    g.add_argument("--init", default="random", choices=init_mod.INITS,
                   help="starting configuration")
    g.add_argument("--jitter", type=float, default=0.0,
                   help="gaussian noise added to the starting configuration")
    g.add_argument("--epsilon", type=float, default=1e-3,
                   help="a spring within this of unit length counts as an edge")

    g = p.add_argument_group("springs")
    g.add_argument("--law", default="gaussian", choices=LAWS,
                   help="how a spring behaves when badly violated")
    g.add_argument("--depth", type=float, default=1.0,
                   help="energy cost of a fully violated spring")
    g.add_argument("--r-min", type=float, default=None,
                   help="hard-core radius keeping balls distinct "
                        "(default: derived from the best known construction at this n)")
    g.add_argument("--k-core", type=float, default=1.0,
                   help="core stiffness, as a multiple of the spring stiffness")
    g.add_argument("--k-conf", type=float, default=2e-3,
                   help="weak spring to the centroid, stops detached balls drifting off")
    g.add_argument("--break-lo", type=float, default=0.0,
                   help="a spring squeezed below this length snaps (0 disables)")
    g.add_argument("--break-hi", type=float, default=float("inf"),
                   help="a spring stretched beyond this length snaps (inf disables)")

    g = p.add_argument_group("annealing")
    g.add_argument("--steps", type=int, default=2500, help="steps per anneal")
    g.add_argument("--sigma0", type=float, default=0.18, help="initial spring range")
    g.add_argument("--sigma1", type=float, default=0.015, help="final spring range")
    g.add_argument("--temp0", type=float, default=0.06, help="initial temperature")
    g.add_argument("--temp1", type=float, default=0.0, help="final temperature")
    g.add_argument("--dt", type=float, default=0.07, help="MD timestep at sigma0")
    g.add_argument("--damping", type=float, default=1.5, help="MD viscous damping at sigma0")
    g.add_argument("--lr", type=float, default=0.05, help="step size for adam / gd")
    g.add_argument("--integrator", default="md", choices=INTEGRATORS,
                   help="md is the literal ball-and-spring dynamics; adam converges harder")

    g = p.add_argument_group("search")
    g.add_argument("--trials", type=int, default=24,
                   help="independent restarts (headless and benchmark)")
    g.add_argument("--cycles", type=int, default=3,
                   help="anneal/project/reheat rounds per restart")
    g.add_argument("--reheat", type=float, default=0.45,
                   help="sigma0 multiplier when restarting from a projected state")
    g.add_argument("--extract-tol", type=float, default=0.02,
                   help="springs this close to unit are handed to the exact solver")
    g.add_argument("--rescue-every", type=int, default=100,
                   help="how often to put balls that snapped every spring back "
                        "into the assembly (0 disables)")

    g = p.add_argument_group("output")
    g.add_argument("--headless", action="store_true", help="no window; run the search")
    g.add_argument("--out", default="results", help="output directory")
    g.add_argument("--no-save", action="store_true", help="do not write result files")
    g.add_argument("--steps-per-frame", type=int, default=8,
                   help="simulation steps between redraws")
    g.add_argument("--band", type=float, default=0.12,
                   help="draw springs this close to unit as faint lines")
    g.add_argument("--candidate", type=float, default=0.02,
                   help="draw springs this close to unit as candidate edges")
    g.add_argument("--benchmark", action="store_true",
                   help="run n=4..14 and compare against the known optima")
    g.add_argument("--quiet", action="store_true", help="less chatter")
    return p


def params_from(args) -> EnergyParams:
    return EnergyParams(
        law=args.law,
        depth=args.depth,
        r_min=default_r_min(args.n) if args.r_min is None else args.r_min,
        k_core=args.k_core,
        k_conf=args.k_conf,
        break_lo=args.break_lo,
        break_hi=args.break_hi,
    )


def schedule_from(args) -> Schedule:
    return Schedule(steps=args.steps, sigma0=args.sigma0, sigma1=args.sigma1,
                    temp0=args.temp0, temp1=args.temp1, dt=args.dt,
                    damping=args.damping, lr=args.lr)


def run_benchmark(args) -> int:
    """Score the solver against every n whose true optimum is known."""
    print(f"{'n':>3}  {'found':>5}  {'known':>5}  {'status':<12}  {'secs':>5}")
    hits = 0
    ns = [n for n in range(4, 15)]
    for n in ns:
        t0 = time.time()
        res = solve_multi(n, trials=args.trials, seed=args.seed, init=args.init,
                          params=EnergyParams(law=args.law, depth=args.depth,
                                              r_min=default_r_min(n),
                                              k_core=args.k_core, k_conf=args.k_conf,
                                              break_lo=args.break_lo,
                                              break_hi=args.break_hi),
                          schedule=schedule_from(args), integrator=args.integrator,
                          epsilon=args.epsilon, cycles=args.cycles, reheat=args.reheat,
                          extract_tol=args.extract_tol, rescue_every=args.rescue_every)
        opt = known_optimum(n)
        if not res.valid:
            # Never let an inflated count be scored as a success.
            status = "AUDIT FAIL"
            print(f"{n:>3}  {res.edges:>5}  {opt:>5}  {status:<12}  "
                  f"{time.time() - t0:>5.1f}")
            for problem in res.problems:
                print(f"       {problem}")
            continue
        if res.edges == opt:
            status, hits = "optimal", hits + 1
        elif res.edges > opt:
            # u(n) is proven for these n, so this can only be a counting bug.
            status = "ABOVE -- BUG"
        else:
            status = f"short by {opt - res.edges}"
        print(f"{n:>3}  {res.edges:>5}  {opt:>5}  {status:<12}  {time.time() - t0:>5.1f}")
    print(f"\nmatched the known optimum in {hits}/{len(ns)} cases")
    return 0 if hits == len(ns) else 1


def run_headless(args) -> int:
    t0 = time.time()

    def progress(trial, res, best):
        if not args.quiet:
            flag = "" if res.valid else "  (rejected: balls too close)"
            print(f"  trial {trial + 1:>3}/{args.trials}  edges {res.edges:>5}"
                  f"  best {best.edges:>5}{flag}")

    res = solve_multi(args.n, trials=args.trials, seed=args.seed, init=args.init,
                      params=params_from(args), schedule=schedule_from(args),
                      integrator=args.integrator, epsilon=args.epsilon,
                      jitter=args.jitter, cycles=args.cycles, reheat=args.reheat,
                      extract_tol=args.extract_tol, rescue_every=args.rescue_every,
                      progress=progress)

    print()
    print(summarise(args.n, res.edges))
    print(f"max edge error {res.max_edge_error:.2e}   min separation "
          f"{res.min_separation:.4f}   max degree {res.max_degree}")
    print(f"audit {'PASS' if res.valid else 'FAIL'}: {res.distinct_points}/"
          f"{args.n} distinct points, recounted from the coordinates alone")
    for problem in res.problems:
        print(f"  - {problem}")
    print(f"{time.time() - t0:.1f}s")

    if not args.no_save:
        os.makedirs(args.out, exist_ok=True)
        stem = os.path.join(args.out, f"n{args.n}_e{res.edges}")
        res.save(stem + ".json")
        try:
            from .render import render_static
            render_static(res.X, res.edge_list, stem + ".png",
                          title=f"n = {args.n}   unit distances = {res.edges}")
            print(f"wrote {stem}.json and {stem}.png")
        except ImportError:
            print(f"wrote {stem}.json  (matplotlib not installed, no PNG)")
    return 0


def run_live(args) -> int:
    try:
        from .render import LiveView
    except ImportError:
        print("matplotlib is required for the live view; "
              "install it or pass --headless")
        return 2

    rng = np.random.default_rng(args.seed)
    counter = {"seed": args.seed}

    def reseed():
        counter["seed"] += 1
        return init_mod.make(args.init, args.n, np.random.default_rng(counter["seed"]),
                             jitter=args.jitter)

    X0 = init_mod.make(args.init, args.n, rng, jitter=args.jitter)
    sim = Simulation(X0, params=params_from(args), schedule=schedule_from(args),
                     integrator=args.integrator, epsilon=args.epsilon, rng=rng,
                     rescue_every=args.rescue_every)

    if not args.quiet:
        print(f"n = {args.n}, {args.n * (args.n - 1) // 2} springs, "
              f"law = {args.law}, integrator = {args.integrator}")
        print("close the window or press q to quit")

    view = LiveView(sim, steps_per_frame=args.steps_per_frame, band=args.band,
                    candidate=args.candidate, on_reseed=reseed, outdir=args.out,
                    title=f"Erdos unit distance -- n = {args.n}")
    view.run()

    print()
    print(summarise(args.n, sim.best_edges))
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.benchmark:
        return run_benchmark(args)
    if args.headless:
        return run_headless(args)
    return run_live(args)


if __name__ == "__main__":
    raise SystemExit(main())
