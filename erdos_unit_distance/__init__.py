"""Numerically attacking the Erdos unit-distance problem with springs.

Every pair of balls is joined by a spring of rest length 1.  Relax the
resulting (heavily frustrated) complete graph, anneal the range over
which a spring stays stiff, and count how many springs end up within
epsilon of unit length.
"""

from .energy import EnergyParams, energy_and_grad, soft_count
from .sim import Schedule, Simulation
from .solver import SolveResult, project, solve_multi, solve_once

__all__ = [
    "EnergyParams", "energy_and_grad", "soft_count",
    "Schedule", "Simulation",
    "SolveResult", "project", "solve_once", "solve_multi",
]
__version__ = "0.1.0"
