"""Random starts, optimised, drawn side by side.

The point of the pictures this writes is *not* the edge counts -- those
are already in the README table.  It is that independent runs from
uniform random points keep landing on the same small set of shapes.  The
solver is never told about any construction; if rhombi and hexagonal
flowers keep appearing anyway, that is the search finding them.

    python scripts/random_gallery.py            # both figures
    python scripts/random_gallery.py --quick    # fewer restarts, for a smoke test

Writes docs/random_gallery.png (best of many restarts, across n) and
docs/random_seeds.png (twelve independent single runs at one n), plus the
raw configurations under results/random_gallery/ so a panel can be
re-examined without re-running the search.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from erdos_unit_distance.counting import audit
from erdos_unit_distance.known import known_optimum
from erdos_unit_distance.solver import solve_multi, solve_once

from make_figures import BG, FG, DOCS, draw_panel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results", "random_gallery")

GALLERY_NS = (7, 9, 12, 14, 16, 20, 24, 30)
SEED_N = 20
SEED_COUNT = 12


def triangles(n: int, edges: np.ndarray) -> int:
    """Unit triangles in the edge set: the cheapest numeric motif proxy.

    A lattice fragment is dense in triangles, a tree or a long chain has
    none, so the count separates "found a patch of the triangular
    lattice" from "found a sprawl" without any eyeballing.
    """
    adj = np.zeros((n, n), dtype=bool)
    adj[edges[:, 0], edges[:, 1]] = True
    adj[edges[:, 1], edges[:, 0]] = True
    return int(sum(np.count_nonzero(adj[i] & adj[j])
                   for i, j in edges)) // 3


def save(res, stem: str) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    res.save(os.path.join(RESULTS, stem + ".json"))


def figure_gallery(path: str, trials: int):
    """Best of `trials` random restarts, at eight sizes."""
    rows = []
    fig, axes = plt.subplots(2, 4, figsize=(21.0, 11.2), facecolor=BG)
    for ax, n in zip(axes.ravel(), GALLERY_NS):
        t0 = time.time()
        res = solve_multi(n, trials=trials, seed=0, init="random")
        checked = audit(res.X, res.edge_list, epsilon=1e-9)
        tri = triangles(n, res.edge_list)
        opt = known_optimum(n)
        mark = "  = u(%d), optimal" % n if opt is not None and res.edges == opt else ""
        draw_panel(ax, res.X, 1e-9, 1e-9,
                   "n = %d   unit distances = %d%s" % (n, res.edges, mark),
                   "%d unit triangles    audit %s, %d/%d distinct"
                   % (tri, "PASS" if checked.ok else "FAIL",
                      checked.distinct_points, n))
        save(res, "n%02d_best" % n)
        rows.append(dict(n=n, edges=res.edges, triangles=tri, optimum=opt,
                         valid=bool(checked.ok), seconds=round(time.time() - t0, 1)))
        print("  n=%-3d edges=%-4d triangles=%-4d %s  %.1fs"
              % (n, res.edges, tri, "PASS" if checked.ok else "FAIL",
                 time.time() - t0))

    fig.suptitle("random start + optimisation, best of %d restarts: "
                 "the same motifs keep coming back" % trials,
                 color=FG, fontsize=13, family="monospace", y=0.99)
    fig.tight_layout(rect=(0, 0.01, 1, 0.95))
    fig.subplots_adjust(hspace=0.22)
    fig.savefig(path, facecolor=BG, dpi=110)
    plt.close(fig)
    print("wrote", path)
    return rows


def figure_seeds(path: str, n: int = SEED_N, count: int = SEED_COUNT):
    """One run per seed, no cherry-picking: the spread and what repeats."""
    rows = []
    fig, axes = plt.subplots(3, 4, figsize=(21.0, 16.2), facecolor=BG)
    for ax, seed in zip(axes.ravel(), range(count)):
        res = solve_once(n, seed=seed, init="random")
        checked = audit(res.X, res.edge_list, epsilon=1e-9)
        tri = triangles(n, res.edge_list)
        draw_panel(ax, res.X, 1e-9, 1e-9,
                   "seed %d   unit distances = %d" % (seed, res.edges),
                   "%d unit triangles    audit %s"
                   % (tri, "PASS" if checked.ok else "FAIL"))
        save(res, "n%02d_seed%02d" % (n, seed))
        rows.append(dict(seed=seed, edges=res.edges, triangles=tri,
                         valid=bool(checked.ok)))
        print("  seed=%-3d edges=%-4d triangles=%-4d %s"
              % (seed, res.edges, tri, "PASS" if checked.ok else "FAIL"))

    counts = [r["edges"] for r in rows]
    fig.suptitle("n = %d, %d independent random starts, every run kept: "
                 "%d-%d unit distances, median %d"
                 % (n, count, min(counts), max(counts), int(np.median(counts))),
                 color=FG, fontsize=13, family="monospace", y=0.99)
    fig.tight_layout(rect=(0, 0.01, 1, 0.96))
    fig.subplots_adjust(hspace=0.22)
    fig.savefig(path, facecolor=BG, dpi=110)
    plt.close(fig)
    print("wrote", path)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=int, default=48,
                   help="restarts behind each gallery panel")
    p.add_argument("--quick", action="store_true", help="a fast smoke test")
    args = p.parse_args()
    trials = 4 if args.quick else args.trials

    print("gallery: best of %d restarts" % trials)
    gallery = figure_gallery(os.path.join(DOCS, "random_gallery.png"), trials)
    print("\nseeds: %d independent single runs at n = %d" % (SEED_COUNT, SEED_N))
    seeds = figure_seeds(os.path.join(DOCS, "random_seeds.png"))

    os.makedirs(RESULTS, exist_ok=True)
    summary = os.path.join(RESULTS, "summary.json")
    with open(summary, "w", encoding="utf-8") as fh:
        json.dump(dict(trials=trials, gallery=gallery,
                       seed_n=SEED_N, seeds=seeds), fh, indent=2)
    print("\nwrote", summary)


if __name__ == "__main__":
    main()
