"""The view must actually draw, on a machine with no display."""

import os

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from erdos_unit_distance import init as I
from erdos_unit_distance.counting import unit_edges
from erdos_unit_distance.render import LiveView, render_static
from erdos_unit_distance.sim import Schedule, Simulation


def test_static_render_writes_a_png(tmp_path):
    X = I.triangular_lattice(12)
    path = tmp_path / "still.png"
    render_static(X, unit_edges(X, 1e-9), str(path), title="test")
    assert path.exists() and path.stat().st_size > 1000


def make_view(n=10, **kw):
    rng = np.random.default_rng(0)
    sim = Simulation(I.make("random", n, rng), schedule=Schedule(steps=300), rng=rng)
    return LiveView(sim, steps_per_frame=5, **kw)


def test_live_view_steps_and_draws():
    view = make_view()
    view.run(max_frames=6)
    assert view.frame == 6
    assert view.sim.step_index == 30


def test_live_view_handles_an_empty_edge_set():
    """Early on nothing is near unit length; drawing must not blow up."""
    view = make_view()
    view.sim.epsilon = 1e-12
    view._draw()


def test_keys_toggle_state():
    view = make_view()
    event = type("E", (), {"key": " "})()
    view._on_key(event)
    assert view.paused
    view._on_key(event)
    assert not view.paused

    eps = view.sim.epsilon
    view._on_key(type("E", (), {"key": "]"})())
    assert view.sim.epsilon > eps
    view._on_key(type("E", (), {"key": "["})())
    assert view.sim.epsilon == pytest.approx(eps)

    view._on_key(type("E", (), {"key": "a"})())
    assert not view.show_faint


def test_project_key_makes_the_edges_exact():
    from erdos_unit_distance.counting import count_unit_distances

    view = make_view(n=12)
    view.run(max_frames=40)
    view._on_key(type("E", (), {"key": "p"})())
    assert count_unit_distances(view.sim.X, 1e-9) >= 1


def test_save_writes_both_files(tmp_path):
    view = make_view()
    view.outdir = str(tmp_path)
    view.run(max_frames=3)
    stem = view.save()
    assert os.path.exists(stem + ".png")
    assert os.path.exists(stem + ".json")


def test_hud_warns_about_coincident_balls():
    """A stacked configuration must say so on screen, not just show a
    flattering edge count."""
    view = make_view(n=9)
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    view.sim.X = np.repeat(corners, 3, axis=0)
    view._draw()
    assert "coincident" in view.hud.get_text()


def test_hud_is_quiet_for_a_healthy_assembly():
    view = make_view(n=12)
    view.sim.X = I.triangular_lattice(12)
    view._draw()
    assert "coincident" not in view.hud.get_text()
