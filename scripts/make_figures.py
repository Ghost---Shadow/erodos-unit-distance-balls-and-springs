"""Regenerate the figures used in the README.

    python scripts/make_figures.py

Writes into docs/.  Everything is seeded, so re-running reproduces the
same pictures.
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from erdos_unit_distance import init as I
from erdos_unit_distance.counting import audit, count_unit_distances, unit_edges
from erdos_unit_distance.energy import EnergyParams
from erdos_unit_distance.known import default_r_min, known_optimum
from erdos_unit_distance.polish import residuals
from erdos_unit_distance.render import LiveView
from erdos_unit_distance.sim import Schedule, Simulation
from erdos_unit_distance.solver import solve_multi

DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
BG = "#0b0e14"
FG = "#c8d3e0"
EDGE = "#5ee2c0"
BALL = "#ffd166"


def segments(X, pairs):
    if pairs.shape[0] == 0:
        return np.zeros((0, 2, 2))
    return np.stack([X[pairs[:, 0]], X[pairs[:, 1]]], axis=1)


WARM = np.array([0.91, 0.52, 0.37])      # spring compressed, pushing apart
COOL = np.array([0.44, 0.66, 0.86])      # spring stretched, pulling together


def faint_colors(strain, band):
    """Warm for compressed, cool for stretched, fading out with strain.

    A diverging colormap would go *white* at zero strain, which reads as
    brighter than the edges it is meant to sit behind.  Encoding closeness
    as opacity instead keeps the bright edge layer on top.
    """
    t = np.clip(np.abs(strain) / band, 0.0, 1.0)
    rgba = np.zeros((strain.size, 4))
    rgba[:, :3] = np.where((strain < 0.0)[:, None], WARM, COOL)
    rgba[:, 3] = 0.10 + 0.45 * (1.0 - t)
    return rgba


def draw_panel(ax, X, epsilon, band, title, subtitle="", dot_size=22,
               subtitle_below=False):
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#232a38")

    if band > epsilon:
        near = unit_edges(X, band)
        if near.shape[0]:
            ax.add_collection(LineCollection(
                segments(X, near), linewidths=0.7, zorder=1,
                colors=faint_colors(residuals(X, near), band)))

    edges = unit_edges(X, epsilon)
    ax.add_collection(LineCollection(segments(X, edges), colors=EDGE,
                                     linewidths=1.7, alpha=0.95, zorder=2))
    ax.scatter(X[:, 0], X[:, 1], s=dot_size, c=BALL, edgecolors="#1b1f2a",
               linewidths=0.5, zorder=3)

    lo, hi = X.min(axis=0), X.max(axis=0)
    cx, cy = 0.5 * (lo + hi)
    half = 0.5 * max(float((hi - lo).max()), 1.0) + 0.35
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_title(title, color=FG, fontsize=11, family="monospace", pad=8)
    if subtitle:
        # In a grid, a subtitle hung below one row collides with the next
        # row's titles, so it goes inside the axes by default.  A single
        # row has the room to put it underneath, clear of the drawing.
        if subtitle_below:
            ax.text(0.5, -0.03, subtitle, transform=ax.transAxes, ha="center",
                    va="top", color="#8b9ab2", fontsize=9.5, family="monospace")
        else:
            ax.text(0.5, 0.015, subtitle, transform=ax.transAxes, ha="center",
                    va="bottom", color="#8b9ab2", fontsize=9, family="monospace")


def figure_anneal(path, n=28, seed=3):
    """One run end to end: blob, negotiation, lock-in, then the exact solve."""
    from erdos_unit_distance.solver import project

    rng = np.random.default_rng(seed)
    params = EnergyParams(r_min=default_r_min(n))
    sim = Simulation(I.make("random", n, rng), params=params,
                     schedule=Schedule(steps=2500), rng=rng)

    frames = []
    for target in (0, 400, 1200, 2500):
        while sim.step_index < target:
            sim.step()
        frames.append((f"step {sim.step_index}   sigma {sim.sigma:.3f}",
                       sim.X.copy(), 0.02, 0.12,
                       f"{count_unit_distances(sim.X, 0.02)} springs within 0.02"))

    Xp, edges = project(sim.X, r_min=params.r_min)
    frames.append(("after the exact solve", Xp, 1e-9, 1e-9,
                   f"{edges.shape[0]} springs at exactly 1"))

    fig, axes = plt.subplots(1, 5, figsize=(22.0, 5.0), facecolor=BG)
    for ax, (title, X, eps, band, sub) in zip(axes, frames):
        draw_panel(ax, X, eps, band, title, sub)
    fig.suptitle(f"one run, n = {n}:  the complete spring graph settling, "
                 f"then snapped exact",
                 color=FG, fontsize=13, family="monospace", y=0.98)
    fig.tight_layout(rect=(0, 0.0, 1, 0.93))
    fig.savefig(path, facecolor=BG, dpi=115)
    plt.close(fig)
    print("wrote", path)


def figure_solutions(path, ns=(7, 12, 14, 20, 24, 30), trials=24):
    """Finished configurations, every drawn segment exactly length 1.

    24 restarts, matching the CLI default.  With spring failure off
    n = 7 needs about that many to find the hexagonal flower, and a
    panel captioned "optimal" had better actually be optimal.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 11.6), facecolor=BG)
    for ax, n in zip(axes.ravel(), ns):
        res = solve_multi(n, trials=trials, seed=0)
        checked = audit(res.X, res.edge_list, epsilon=1e-9)
        opt = known_optimum(n)
        note = f"optimal: u({n}) = {opt}" if opt is not None and res.edges == opt \
            else f"max edge error {res.max_edge_error:.0e}"
        # State the overlap check on the picture itself: a high count means
        # nothing without it.
        note += f"    audit {'PASS' if checked.ok else 'FAIL'}, "
        note += f"{checked.distinct_points}/{n} points distinct"
        draw_panel(ax, res.X, 1e-9, 1e-9,
                   f"n = {n}   unit distances = {res.edges}", note)
        print(f"  n={n} edges={res.edges} audit={'PASS' if checked.ok else 'FAIL'}")
    fig.suptitle("solved configurations: every segment is exactly length 1",
                 color=FG, fontsize=13, family="monospace", y=0.99)
    fig.tight_layout(rect=(0, 0.01, 1, 0.95))
    fig.subplots_adjust(hspace=0.22)
    fig.savefig(path, facecolor=BG, dpi=110)
    plt.close(fig)
    print("wrote", path)


def figure_constructions(path, n=150):
    """The lattice-product family on both base lattices.

    Columns 1-4 are a basis Minkowski-summed with a unit lattice, whose
    density limit is z/2 + u(B)/|B|.  Column 5 is the same base lattice
    with Erdos' rescaling trick applied instead -- not a basis product at
    all, but the natural thing to compare against, since it is the other
    way of getting more unit distances out of the same lattice.

    The two rows differ only in the base: a 60-degree rhombus has a unit
    short diagonal on top of its two sides, so the rhombic row starts at
    3 edges per point where the square row starts at 2.
    """
    from erdos_unit_distance.counting import min_separation

    NL = chr(10)
    seg = np.array([[0.0, 0.0], [1.0, 0.0]])
    bases = [("1 point", np.zeros((1, 2))),
             ("segment", seg),
             ("triangle", I.unit_triangle()),
             ("centered hexagon", I.centered_hexagon())]
    rescaled = {"triangular": ("Eisenstein grid", I.eisenstein_grid),
                "square": ("Erdos grid", I.erdos_grid)}

    fig, axes = plt.subplots(2, 5, figsize=(23.5, 10.6), facecolor=BG)
    for row, lattice in enumerate(("triangular", "square")):
        base_name = "rhombic" if lattice == "triangular" else "square"
        for col, (label, B) in enumerate(bases):
            ax = axes[row, col]
            P = I.lattice_product(n, B, lattice=lattice)
            edges = count_unit_distances(P, 1e-9)
            limit = I.density_limit(B, lattice)
            ok = audit(P, epsilon=1e-9).ok
            title = NL.join([
                base_name + " base  x  " + label,
                "%d edges = %.2fn -> %.2fn" % (edges, edges / P.shape[0], limit)])
            sub = NL.join(["sep %.2f" % min_separation(P),
                           "PASS" if ok else "FAIL"])
            draw_panel(ax, P, 1e-9, 1e-9, title, sub, dot_size=9)
            print("  %10s x %-17s %5d edges  %.2fn (limit %.2fn)"
                  % (base_name, label, edges, edges / P.shape[0], limit))

        # Fifth column: the rescaling trick on the same base lattice.
        ax = axes[row, 4]
        name, fn = rescaled[lattice]
        P = fn(n)
        edges = count_unit_distances(P, 1e-9)
        ok = audit(P, epsilon=1e-9).ok
        title = NL.join([base_name + " base  +  rescaling",
                         "%d edges = %.2fn" % (edges, edges / P.shape[0])])
        sub = NL.join([name, "PASS" if ok else "FAIL"])
        draw_panel(ax, P, 1e-9, 1e-9, title, sub, dot_size=9)
        print("  %10s + rescaling        %5d edges  %.2fn  (%s)"
              % (base_name, edges, edges / P.shape[0], name))

    fig.suptitle("basis x lattice (density -> z/2 + u(B)/|B|), and the same "
                 "lattices under Erdos rescaling",
                 color=FG, fontsize=13, family="monospace", y=0.985)
    fig.tight_layout(rect=(0, 0.01, 1, 0.955))
    fig.subplots_adjust(hspace=0.20)
    fig.savefig(path, facecolor=BG, dpi=110)
    plt.close(fig)
    print("wrote", path)


def figure_growth(path):
    """Counts and densities against n, with the known bounds for scale.

    The construction gallery only covers the B (+) lattice family, so the
    Erdos rescaled grid never appears there -- it is a different animal.
    Here it does, next to the O(n^4/3) upper bound that caps everything.
    """
    seg = np.array([[0.0, 0.0], [1.0, 0.0]])
    families = [
        ("square lattice", lambda n: I.square_lattice(n)),
        ("triangular lattice", lambda n: I.triangular_lattice(n)),
        ("prism of triangles", lambda n: I.prism_lattice(n)),
        ("prism doubled", lambda n: I.prism_double(n)),
        ("centered hexagon x lattice", lambda n: I.hex_lattice(n)),
        ("Erdos rescaled grid", lambda n: I.erdos_grid(n)),
        ("Eisenstein grid", lambda n: I.eisenstein_grid(n)),
    ]

    data = {}
    for label, fn in families:
        pts = [(n, count_unit_distances(fn(n), 1e-9)) for n in GROWTH_SIZES]
        data[label] = pts
        print(f"  {label:<28} " + " ".join(f"{e:>6}" for _, e in pts))

    # Flower sums only hold their structure at n = 7^k, so they get their
    # own sample points rather than being forced onto the shared grid.
    flower = [(7 ** k, count_unit_distances(I.hex_minkowski(7 ** k), 1e-9))
              for k in range(2, 6)]
    data["rotated flower sums"] = flower
    print("  rotated flower sums          " + " ".join(f"{e:>6}" for _, e in flower))

    # The solver itself, as a baseline.  It only reaches small n -- each
    # point is a full multi-restart search -- and that ceiling is exactly
    # the thing worth showing next to the constructions.
    #
    # Cached: recomputing this on every figure rebuild costs minutes and
    # is the slowest thing in this script by a wide margin.  Delete the
    # cache file to re-measure.
    import json

    cache = os.path.join(DOCS, "solver_baseline.json")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as fh:
            solver = [tuple(row) for row in json.load(fh)]
        print(f"  random + optimisation        (cached from {cache})")
    else:
        from erdos_unit_distance.solver import solve_multi
        solver = [(n, solve_multi(n, trials=4, seed=0).edges)
                  for n in (20, 40, 80)]
        with open(cache, "w", encoding="utf-8") as fh:
            json.dump(solver, fh)
    data["random + optimisation"] = solver
    print("  random + optimisation        " +
          " ".join(f"{e:>6}" for _, e in solver))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.0, 7.0), facecolor=BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_color("#2a3244")
        ax.tick_params(colors="#8b9ab2", labelsize=9)
        ax.grid(True, which="both", color="#1b2130", lw=0.7)
        ax.set_axisbelow(True)

    # -- counts, log-log, with the bounds ----------------------------- #
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    grid_pts = data["Erdos rescaled grid"]
    n_hi, e_hi = grid_pts[-1]

    ns = np.logspace(np.log10(15), np.log10(20000), 60)
    # The upper bound's constant is not determined; anchor it above the
    # best construction so it reads as an envelope, not a prediction.
    ax1.plot(ns, 2.2 * e_hi * (ns / n_hi) ** (4.0 / 3.0), color="#8b9ab2",
             lw=1.6, ls=(0, (6, 4)), zorder=1)
    ax1.text(ns[-1], 2.2 * e_hi * (ns[-1] / n_hi) ** (4.0 / 3.0),
             "  O(n^4/3) upper bound", color="#8b9ab2", fontsize=9.5,
             family="monospace", va="center")
    ax1.text(ns[-1], 1.05 * e_hi * (ns[-1] / n_hi) ** (4.0 / 3.0),
             "  Spencer-Szemeredi-Trotter 1984", color="#6b7a90",
             fontsize=8.5, family="monospace", va="center")
    ax1.plot(ns, 3.0 * ns, color="#3d4a5c", lw=1.4, ls=(0, (2, 3)), zorder=1)
    ax1.text(ns[-1], 3.0 * ns[-1], "  3n", color="#3d4a5c", fontsize=9.5,
             family="monospace", va="center")

    for label, pts in data.items():
        xs = [n for n, _ in pts]
        ys = [e for _, e in pts]
        shown = {"Erdos rescaled grid": "Erdos grid: n^(1+c/log log n), Erdos 1946",
                 "random + optimisation": "random + optimisation (this solver)",
                 }.get(label, label)
        # The solver is dashed: it is a search result, not a construction,
        # and it stops early because each point costs a full search.
        style = "--s" if label == "random + optimisation" else "-o"
        ax1.plot(xs, ys, style, color=GROWTH_COLORS[label], lw=1.9, ms=4.5,
                 label=shown, zorder=4 if label == "random + optimisation" else 3)
    ax1.set_xlabel("n (points)", color=FG, fontsize=10, family="monospace")
    ax1.set_ylabel("unit distances", color=FG, fontsize=10, family="monospace")
    ax1.set_title("counts, log-log", color=FG, fontsize=12,
                  family="monospace", pad=10)
    ax1.set_xlim(15, 60000)
    leg = ax1.legend(loc="lower right", fontsize=9, facecolor="#11151f",
                     edgecolor="#2a3244", labelcolor=FG, framealpha=0.95)
    for text in leg.get_texts():
        text.set_family("monospace")

    # -- density, the discriminating view ----------------------------- #
    ax2.set_xscale("log")
    for label, pts in data.items():
        xs = [n for n, _ in pts]
        ys = [e / n for n, e in pts]
        style = "--s" if label == "random + optimisation" else "-o"
        ax2.plot(xs, ys, style, color=GROWTH_COLORS[label], lw=1.9, ms=4.5,
                 zorder=4 if label == "random + optimisation" else 3)
    # 4.5 and 4.71 are too close to label side by side, so the doubled
    # line is drawn but named in the caption instead.
    for y, text in ((2.0, "2n square"), (3.0, "3n triangular"),
                    (4.0, "4n prism"), (4.5, ""),
                    (3.0 + 12.0 / 7.0, "4.5n doubled / 4.71n hexagon")):
        ax2.axhline(y, color="#3d4a5c", lw=0.9, ls=(0, (2, 3)), zorder=1)
        if text:
            ax2.text(17.0, y + 0.13, text, color="#6b7a90", fontsize=8.5,
                     family="monospace", va="bottom", ha="left")
    ax2.set_xlabel("n (points)", color=FG, fontsize=10, family="monospace")
    ax2.set_ylabel("unit distances / n", color=FG, fontsize=10,
                   family="monospace")
    ax2.set_title("density: the linear families flatten, the others do not",
                  color=FG, fontsize=12, family="monospace", pad=10)
    ax2.set_xlim(15, 3.0e4)
    ax2.set_ylim(1.5, 10.5)

    fig.suptitle("growth rates: u(n) lies between the Erdos construction "
                 "and the Spencer-Szemeredi-Trotter bound",
                 color=FG, fontsize=13, family="monospace", y=0.975)
    fig.tight_layout(rect=(0, 0.01, 1, 0.93))
    fig.savefig(path, facecolor=BG, dpi=120)
    plt.close(fig)
    print("wrote", path)




def _loeschian_factors(r):
    """Factor r and mark the primes congruent to 1 mod 3.

    Those are the ones that split in the Eisenstein integers, and each of
    them multiplies the number of ways r can be written as i^2 + ij + j^2.
    """
    r = int(r)
    out, d = [], 2
    while d * d <= r:
        while r % d == 0:
            out.append(d)
            r //= d
        d += 1
    if r > 1:
        out.append(r)
    return out


def figure_eisenstein(path):
    """What the Eisenstein grid actually is, and why it works.

    It is the plain triangular lattice, rescaled so that its *most
    frequently realised* distance becomes 1.  The point set is unchanged;
    what changes is which pairs count.
    """
    NL = chr(10)
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 6.6), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_color("#232a38")

    # -- panel 1: the distance spectrum, i.e. the spike being aimed at --
    ax = axes[0]
    n_spec = 900
    rows = I.distance_spectrum(I.triangular_lattice(n_spec), top=14)
    mults = [m for m, _ in rows]
    norms = [int(round(d * d)) for _, d in rows]
    colors = ["#c678dd" if i == 0 else "#3d4a5c" for i in range(len(rows))]
    ax.bar(range(len(rows)), mults, color=colors, width=0.75)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([str(r) for r in norms], color="#8b9ab2",
                       fontsize=8.5, family="monospace")
    ax.tick_params(colors="#8b9ab2", labelsize=8.5)
    ax.set_xlabel("squared distance  r = i^2 + ij + j^2", color=FG,
                  fontsize=10, family="monospace")
    ax.set_ylabel("pairs realising it", color=FG, fontsize=10,
                  family="monospace")
    ax.set_title("the spectrum has a spike to aim at", color=FG,
                 fontsize=12, family="monospace", pad=10)
    win_r, win_m = norms[0], mults[0]
    facs = _loeschian_factors(win_r)
    ax.text(0.5, 0.94, "n = %d: winner r = %d = %s" %
            (n_spec, win_r, " x ".join(str(f) for f in facs)),
            transform=ax.transAxes, ha="center", color="#c678dd",
            fontsize=10.5, family="monospace")
    ax.text(0.5, 0.88, "every factor is 1 mod 3, and each one multiplies",
            transform=ax.transAxes, ha="center", color="#8b9ab2",
            fontsize=9.5, family="monospace")
    ax.text(0.5, 0.83, "the number of representations", transform=ax.transAxes,
            ha="center", color="#8b9ab2", fontsize=9.5, family="monospace")

    # -- panel 2: the grid itself, small enough to read ----------------
    ax = axes[1]
    n_small = 120
    P = I.eisenstein_grid(n_small)
    edges = unit_edges(P, 1e-9)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.add_collection(LineCollection(segments(P, edges), colors=EDGE,
                                     linewidths=1.0, alpha=0.75, zorder=2))
    ax.scatter(P[:, 0], P[:, 1], s=16, c=BALL, edgecolors="#1b1f2a",
               linewidths=0.5, zorder=3)
    lo, hi = P.min(axis=0), P.max(axis=0)
    cx, cy = 0.5 * (lo + hi)
    half = 0.5 * float((hi - lo).max()) + 0.3
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    small_r = int(round(I.distance_spectrum(I.triangular_lattice(n_small), 1)[0][1] ** 2))
    ax.set_title(NL.join(["the same points, rescaled so r = %d is unit" % small_r,
                          "n = %d, %d unit distances = %.2fn"
                          % (n_small, edges.shape[0], edges.shape[0] / n_small)]),
                 color=FG, fontsize=12, family="monospace", pad=10)

    # -- panel 3: one ball's neighbourhood ------------------------------
    ax = axes[2]
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    centre = int(np.argmin(((P - P.mean(axis=0)) ** 2).sum(axis=1)))
    d = np.sqrt(((P - P[centre]) ** 2).sum(axis=1))
    nb = np.nonzero(np.abs(d - 1.0) < 1e-9)[0]
    ax.scatter(P[:, 0], P[:, 1], s=14, c="#3d4a5c", zorder=1)
    for j in nb:
        ax.plot([P[centre, 0], P[j, 0]], [P[centre, 1], P[j, 1]],
                color="#c678dd", lw=1.8, zorder=2)
    ax.scatter(P[nb, 0], P[nb, 1], s=42, c=EDGE, edgecolors="#1b1f2a",
               linewidths=0.6, zorder=3)
    ax.scatter([P[centre, 0]], [P[centre, 1]], s=80, c=BALL,
               edgecolors="#1b1f2a", linewidths=0.8, zorder=4)
    ax.set_xlim(P[centre, 0] - 1.6, P[centre, 0] + 1.6)
    ax.set_ylim(P[centre, 1] - 1.6, P[centre, 1] + 1.6)
    # r = 7 splits in the Eisenstein integers, so its shell holds 12
    # points: the 6 lattice directions rotated both ways off axis.
    ax.set_title(NL.join(["one ball's unit neighbours: %d of them" % nb.size,
                          "the r = %d shell, reaching past the 6 nearest"
                          % small_r]),
                 color=FG, fontsize=12, family="monospace", pad=10)

    fig.suptitle("the Eisenstein grid: a triangular lattice rescaled to its "
                 "most popular distance",
                 color=FG, fontsize=13, family="monospace", y=0.98)
    fig.tight_layout(rect=(0, 0.01, 1, 0.92))
    fig.savefig(path, facecolor=BG, dpi=120)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
