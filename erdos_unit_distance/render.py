"""Live 2D view of the ball-and-spring assembly (matplotlib).

Four layers are drawn on top of each other:

* every spring currently *near* unit length, faint, warm where compressed
  and cool where stretched -- the springs still negotiating;
* the springs within ``candidate`` of unit, dim green -- what the
  relaxation has agreed on and will hand to the exact solver;
* the springs within ``epsilon`` of unit, bright -- the edges, the thing
  actually being counted;
* the balls themselves.

The candidate layer exists because the anneal only tightens springs to
about ``sigma1``: at the strict epsilon almost nothing lights up until the
projection runs, and the window looks empty for most of a run.  Watching
the faint set condense into the candidate set, and that into the bright
set, is the whole story of the method in one picture.
"""

from __future__ import annotations

import numpy as np

from .counting import merge_duplicates, min_separation, unit_edges
from .polish import residuals
from .solver import project

HELP = """\
space  pause / resume        r  restart this configuration
n      new random start      p  project onto exact unit distances now
[ / ]  epsilon down / up     a  toggle the faint near-unit springs
s      save PNG + JSON       q  quit
bright = exact edges   dim green = candidates for the exact solver
"""

_BG = "#0b0e14"
_FG = "#c8d3e0"
_EDGE = "#5ee2c0"
_CAND = "#2f7f6d"
_BALL = "#ffd166"
_WARM = np.array([0.91, 0.52, 0.37])     # spring compressed, pushing apart
_COOL = np.array([0.44, 0.66, 0.86])     # spring stretched, pulling together


def _faint_colors(strain: np.ndarray, band: float) -> np.ndarray:
    """Warm for compressed, cool for stretched, fading out with strain.

    A diverging colormap would go *white* at zero strain, which reads as
    brighter than the bright edge layer it sits behind.  Encoding
    closeness as opacity instead keeps the layering honest.
    """
    t = np.clip(np.abs(strain) / max(band, 1e-12), 0.0, 1.0)
    rgba = np.zeros((strain.size, 4))
    rgba[:, :3] = np.where((strain < 0.0)[:, None], _WARM, _COOL)
    rgba[:, 3] = 0.10 + 0.45 * (1.0 - t)
    return rgba


def _segments(X: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """(m, 2, 2) line segments for a list of index pairs."""
    if pairs.shape[0] == 0:
        return np.zeros((0, 2, 2))
    return np.stack([X[pairs[:, 0]], X[pairs[:, 1]]], axis=1)


class LiveView:
    """Drive a :class:`~erdos_unit_distance.sim.Simulation` in a window."""

    def __init__(self, sim, steps_per_frame: int = 8, band: float = 0.12,
                 candidate: float = 0.02, max_faint: int = 4000, title: str = "",
                 on_reseed=None, outdir: str = "results"):
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        self.sim = sim
        self.steps_per_frame = steps_per_frame
        self.band = band
        # The anneal only tightens springs to ~sigma1, so at the strict
        # epsilon almost nothing lights up until the exact projection runs.
        # This middle layer shows what the relaxation has actually agreed
        # on -- the springs that will be handed to the exact solver.
        self.candidate = candidate
        self.max_faint = max_faint
        self.on_reseed = on_reseed
        self.outdir = outdir
        self.paused = False
        self.show_faint = True
        self.running = True
        self.frame = 0
        self._plt = plt
        # Agg and friends have no window to show or event loop to pump.
        self._interactive = matplotlib.get_backend().lower() not in (
            "agg", "pdf", "ps", "svg", "template")

        plt.rcParams["toolbar"] = "none"
        self.fig, self.ax = plt.subplots(figsize=(9.0, 9.0), facecolor=_BG)
        self.fig.canvas.manager.set_window_title(title or "Erdos unit distance -- springs")
        self.ax.set_facecolor(_BG)
        self.ax.set_aspect("equal")
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

        self.faint = LineCollection([], linewidths=0.8, zorder=1)
        self.mid = LineCollection([], colors=_CAND, linewidths=1.2, alpha=0.75, zorder=2)
        self.bright = LineCollection([], colors=_EDGE, linewidths=2.1, alpha=1.0, zorder=3)
        self.ax.add_collection(self.faint)
        self.ax.add_collection(self.mid)
        self.ax.add_collection(self.bright)
        self.dots = self.ax.scatter([], [], s=34, c=_BALL, edgecolors="#1b1f2a",
                                    linewidths=0.8, zorder=3)

        self.hud = self.ax.text(
            0.015, 0.985, "", transform=self.ax.transAxes, va="top", ha="left",
            color=_FG, fontsize=10.5, family="monospace", zorder=4,
            bbox=dict(facecolor="#11151f", edgecolor="#2a3244", pad=7.0, alpha=0.92))
        self.keys = self.ax.text(
            0.015, 0.015, HELP, transform=self.ax.transAxes, va="bottom", ha="left",
            color="#6b7a90", fontsize=8.5, family="monospace", zorder=4)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", lambda _e: setattr(self, "running", False))

    # ------------------------------------------------------------------ #

    def _on_key(self, event):
        key = (event.key or "").lower()
        if key in ("q", "escape"):
            self.running = False
            self._plt.close(self.fig)
        elif key == " ":
            self.paused = not self.paused
        elif key == "r":
            self.sim.reset()
        elif key == "n" and self.on_reseed is not None:
            self.sim.reset(self.on_reseed())
        elif key == "a":
            self.show_faint = not self.show_faint
        elif key == "]":
            self.sim.epsilon = min(self.sim.epsilon * 2.0, 0.5)
        elif key == "[":
            self.sim.epsilon = max(self.sim.epsilon * 0.5, 1e-9)
        elif key == "p":
            X, _ = project(self.sim.X, r_min=self.sim.params.r_min)
            self.sim.X = X
            self.sim.V[:] = 0.0
        elif key == "s":
            self.save()

    def save(self) -> str:
        """Write a PNG of the current view plus the configuration as JSON."""
        import json
        import os

        os.makedirs(self.outdir, exist_ok=True)
        stem = os.path.join(self.outdir, f"n{self.sim.n}_step{self.sim.step_index}")
        self.fig.savefig(stem + ".png", facecolor=_BG, dpi=160)
        edges = unit_edges(self.sim.X, self.sim.epsilon)
        with open(stem + ".json", "w", encoding="utf-8") as fh:
            json.dump({"n": self.sim.n, "epsilon": self.sim.epsilon,
                       "edges": edges.tolist(), "X": self.sim.X.tolist()}, fh, indent=2)
        print(f"saved {stem}.png and {stem}.json")
        return stem

    # ------------------------------------------------------------------ #

    def _draw(self):
        sim = self.sim
        X = sim.X
        eps = sim.epsilon

        edges = unit_edges(X, eps)
        self.bright.set_segments(_segments(X, edges))

        if self.show_faint:
            near = unit_edges(X, max(self.band, eps))
            if near.shape[0] > self.max_faint:            # keep big n drawable
                near = near[np.random.default_rng(0).choice(
                    near.shape[0], self.max_faint, replace=False)]
            self.faint.set_segments(_segments(X, near))
            if near.shape[0]:
                self.faint.set_color(
                    _faint_colors(residuals(X, near), max(self.band, eps)))
        else:
            self.faint.set_segments([])

        # Candidates: what the relaxation has agreed on and will hand to
        # the exact solver.  Without this layer the window looks empty
        # for most of a run, since the anneal only reaches ~sigma1.
        if self.candidate > eps:
            candidates = unit_edges(X, self.candidate)
        else:
            candidates = np.zeros((0, 2), dtype=np.int64)
        n_candidate = int(candidates.shape[0])
        self.mid.set_segments(_segments(X, candidates))

        self.dots.set_offsets(X)

        # Frame the assembly with a little breathing room.
        lo, hi = X.min(axis=0), X.max(axis=0)
        pad = 0.08 * max(float((hi - lo).max()), 1.0) + 0.35
        cx, cy = 0.5 * (lo + hi)
        half = 0.5 * max(float((hi - lo).max()), 1.0) + pad
        self.ax.set_xlim(cx - half, cx + half)
        self.ax.set_ylim(cy - half, cy + half)

        # Coincident balls would let one location be counted many times,
        # so say so loudly rather than quietly reporting a big number.
        sep = min_separation(X)
        distinct = merge_duplicates(X)[0].shape[0] if sep < 1e-4 else sim.n
        duplicate_note = ("" if distinct == sim.n else
                          f"\nWARNING {sim.n - distinct} coincident ball(s): "
                          f"count is inflated")

        s = sim.stats
        self.hud.set_text(
            f"n {sim.n}   step {sim.step_index}/{sim.schedule.steps}"
            f"{'   [paused]' if self.paused else ''}\n"
            f"edges      |d-1| <= {eps:<7.1e} {edges.shape[0]:>5d}\n"
            f"candidates |d-1| <= {self.candidate:<7.3f} {n_candidate:>5d}\n"
            f"best so far                    {sim.best_edges:>5d}\n"
            f"soft count (sigma window)   {s.soft_edges:>8.1f}\n"
            f"sigma  {sim.sigma:.4f}   temp {s.temperature:.4f}\n"
            f"energy {s.energy:9.3f}   |grad| {s.grad_norm:8.3f}\n"
            f"min separation              {sep:>8.3f}\n"
            f"law {sim.params.law}  integrator {sim.integrator}"
            + duplicate_note)

    def run(self, max_frames: int | None = None):
        """Block, stepping and redrawing until the window closes.

        ``max_frames`` stops after that many frames instead, which is how
        the view gets exercised in tests without a human closing a window.
        """
        plt = self._plt
        if self._interactive:
            plt.ion()
            self.fig.show()
        while self.running and plt.fignum_exists(self.fig.number):
            if max_frames is not None and self.frame >= max_frames:
                break
            if not self.paused:
                for _ in range(self.steps_per_frame):
                    if self.sim.done:
                        break
                    self.sim.step()
            self._draw()
            self.frame += 1
            self.fig.canvas.draw_idle()
            if self._interactive:
                self.fig.canvas.flush_events()
                plt.pause(0.001)
        if self._interactive:
            plt.ioff()
        return self.sim


def render_static(X: np.ndarray, edges: np.ndarray, path: str, title: str = "",
                  dpi: int = 160) -> str:
    """Save a still picture of a finished configuration."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig, ax = plt.subplots(figsize=(8.0, 8.0), facecolor=_BG)
    ax.set_facecolor(_BG)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.add_collection(LineCollection(_segments(X, edges), colors=_EDGE,
                                     linewidths=1.9, alpha=0.95))
    ax.scatter(X[:, 0], X[:, 1], s=38, c=_BALL, edgecolors="#1b1f2a",
               linewidths=0.8, zorder=3)
    lo, hi = X.min(axis=0), X.max(axis=0)
    cx, cy = 0.5 * (lo + hi)
    half = 0.5 * max(float((hi - lo).max()), 1.0) + 0.4
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    if title:
        ax.set_title(title, color=_FG, fontsize=12, family="monospace")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.94, bottom=0.03)
    fig.savefig(path, facecolor=_BG, dpi=dpi)
    plt.close(fig)
    return path
