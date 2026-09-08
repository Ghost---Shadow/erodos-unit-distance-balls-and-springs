"""End-to-end behaviour, scored against values that are actually known."""

import numpy as np
import pytest

from erdos_unit_distance import init as I
from erdos_unit_distance.counting import count_unit_distances, min_separation
from erdos_unit_distance.known import MAX_UNIT_DISTANCES, default_r_min, known_optimum
from erdos_unit_distance.sim import Schedule, Simulation
from erdos_unit_distance.solver import project, solve_multi, solve_once


@pytest.mark.parametrize("n,trials", [(6, 10), (7, 24), (9, 10)])
def test_reaches_the_known_optimum(n, trials):
    """n = 7 needs more restarts than the others with spring failure off:
    the hexagonal flower is a narrow basin, and the hard cutoff used to
    help find it."""
    res = solve_multi(n, trials=trials, seed=0)
    assert res.edges == known_optimum(n)


@pytest.mark.parametrize("n", sorted(MAX_UNIT_DISTANCES))
def test_never_exceeds_a_proven_optimum(n):
    """u(n) is proven for these n, so beating it means the count is wrong."""
    if n < 3:
        pytest.skip("trivial")
    res = solve_once(n, seed=0, cycles=1)
    assert res.edges <= known_optimum(n)


@pytest.mark.parametrize("n", [8, 12, 20])
def test_reported_edges_are_exactly_unit_length(n):
    res = solve_once(n, seed=1)
    assert res.max_edge_error < 1e-9
    assert res.edges == count_unit_distances(res.X, 1e-9)


@pytest.mark.parametrize("n", [8, 12, 20])
def test_balls_stay_distinct(n):
    res = solve_once(n, seed=1)
    assert res.valid
    assert res.min_separation > 0.5 * default_r_min(n)


def test_projection_makes_a_noisy_lattice_exact():
    rng = np.random.default_rng(0)
    X = I.triangular_lattice(19) + rng.normal(scale=0.02, size=(19, 2))
    Xp, edges = project(X, r_min=0.4)
    assert edges.shape[0] >= count_unit_distances(I.triangular_lattice(19), 1e-9)
    assert min_separation(Xp) > 0.9


def test_result_round_trips_through_json(tmp_path):
    import json

    res = solve_once(8, seed=0, cycles=1)
    path = tmp_path / "r.json"
    res.save(str(path))
    loaded = json.loads(path.read_text())
    assert loaded["edges"] == res.edges
    assert np.allclose(np.array(loaded["X"]), res.X)


def test_more_cycles_never_hurt():
    """Basin hopping keeps the best result it has seen, so it is monotone."""
    one = solve_once(12, seed=2, cycles=1).edges
    three = solve_once(12, seed=2, cycles=3).edges
    assert three >= one


@pytest.mark.parametrize("integrator", ["md", "adam", "gd"])
def test_every_integrator_runs_and_conserves_the_ball_count(integrator):
    rng = np.random.default_rng(0)
    sim = Simulation(I.make("random", 12, rng), schedule=Schedule(steps=200),
                     integrator=integrator, rng=rng)
    sim.run()
    assert sim.X.shape == (12, 2)
    assert np.isfinite(sim.X).all()


def test_annealing_reduces_the_energy_of_the_starting_blob():
    rng = np.random.default_rng(0)
    sim = Simulation(I.make("random", 15, rng), schedule=Schedule(steps=800), rng=rng)
    start = sim.energy
    sim.run()
    assert sim.energy < start


def test_reset_restores_the_starting_configuration():
    rng = np.random.default_rng(0)
    X0 = I.make("random", 10, rng)
    sim = Simulation(X0, schedule=Schedule(steps=100), rng=rng)
    sim.run()
    sim.reset()
    assert np.allclose(sim.X, X0)
    assert sim.step_index == 0


def test_detached_balls_are_recycled():
    """Spring failure lets a ball drift out of reach of everything; it must
    not be left there doing nothing."""
    rng = np.random.default_rng(0)
    X = I.make("random", 10, rng)
    X[0] = [80.0, 80.0]                       # far past break_hi from anything
    sim = Simulation(X, schedule=Schedule(steps=200), rng=rng, rescue_every=50)
    sim.run()
    assert sim.rescued >= 1
    assert np.linalg.norm(sim.X[0] - sim.X.mean(axis=0)) < 20.0


def test_rescue_can_be_disabled():
    rng = np.random.default_rng(0)
    X = I.make("random", 10, rng)
    X[0] = [80.0, 80.0]
    sim = Simulation(X, schedule=Schedule(steps=200), rng=rng, rescue_every=0)
    sim.run()
    assert sim.rescued == 0


def test_rescue_leaves_a_healthy_assembly_alone():
    rng = np.random.default_rng(0)
    sim = Simulation(I.triangular_lattice(12), schedule=Schedule(steps=50), rng=rng)
    assert sim.rescue_detached() == 0
