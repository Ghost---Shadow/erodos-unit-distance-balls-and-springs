# Erdős unit distance — balls and springs

A numerical attack on the Erdős unit-distance problem: *among `n` points in the
plane, how many pairs can be at distance exactly 1?*

The model is a **complete graph of springs, every one with rest length 1**. All
`n(n-1)/2` pairs pull or push toward unit separation at once, which is wildly
over-constrained — the assembly is frustrated and cannot satisfy everything.
Relax it, anneal, and count the springs that ended up within `ε` of unit length.
Those are the edges of a unit-distance graph.

```bash
pip install -r requirements.txt

python run.py --n 24                       # live 2D view
python run.py --n 40 --headless --trials 24
python run.py --benchmark                  # score against the known optima
```

![solved configurations](./docs/solutions.png)

Every segment above is exactly length 1, to about 2e-16.

## The result that matters

For `n ≤ 14` the true maximum `u(n)` is known by exhaustive search. The solver
reaches it in **every one of those cases** (24 restarts each, the default):

| n | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|----|----|----|----|----|
| known `u(n)` | 5 | 7 | 9 | 12 | 14 | 18 | 20 | 23 | 27 | 30 | 33 |
| found | 5 | 7 | 9 | 12 | 14 | 18 | 20 | 23 | 27 | 30 | 33 |

It also never *exceeds* them, which is the more important check — any value
above `u(n)` would mean the counting is lying rather than the search winning.

**Counts are audited, not trusted.** The cheapest way to fake a high score is to
stop using `n` distinct points: stack 12 balls on 3 spots a unit apart and every
cross-cluster pair "is" a unit distance, scoring 48 against a true optimum of
27. The optimiser finds that immediately if you let it. So every configuration
is re-checked from its coordinates alone, by code that has no idea how it was
produced, and a restart that fails the audit always loses to one that passes.
[Details](./docs/method.md#counts-are-audited-not-trusted).

## How a run works

![one run, start to finish](./docs/anneal.png)

Two halves, because the relaxation is good at combinatorics and bad at
precision — it will happily park a spring at 1.0004.

1. **Anneal** the complete spring graph under damped molecular dynamics,
   narrowing the range `σ` over which a spring stays stiff. This decides *which*
   pairs want to be unit.
2. **Project**: freeze that edge set and solve the geometry exactly with
   Levenberg–Marquardt, driving residuals to ~1e-15, then try to grow it.

Then **reheat** and repeat. That basin hopping is where most of the quality
comes from — extra annealing steps stop paying off around 2500, because what
limits a run is which basin it fell into, not how carefully it descended.

Getting this to work at all needed three non-obvious fixes, each found by
results coming out wrong: saturating springs so far pairs cannot dominate, a
well depth independent of `σ` so the objective does not evaporate as it anneals,
and a hard core stiff enough to survive both.
[The method in full](./docs/method.md).

## How good is it?

Excellent for small `n`, and it runs out of steam quickly.

| n | triangular | Erdős grid | Eisenstein grid | prism | this solver |
|---|---|---|---|---|---|
| 20 | 44 | 37 | 44 | 40 | **48** |
| 30 | 69 | 66 | 79 | 78 | **79** |
| 50 | 123 | 130 | **167** | 135 | 124 |
| 80 | 207 | 222 | **311** | 247 | 171 |

The solver wins at `n = 20`, only ties the best construction at `n = 30`, and is
well behind by `n = 50`. Past that range the analytic families win outright and
random restarts stop finding anything better, because the search space grows far
faster than the restart count. Seeding from a construction and letting it
improve locally is the useful move there:

```bash
python run.py --n 30 --headless --init prism --jitter 0.10 --trials 24
```

(Solver figures use 24 restarts up to `n = 50` and 4 at `n = 80`, so the last is
a lower bound on what it would find given more compute, not a ceiling.)

## Constructions

![construction family](./docs/constructions.png)

Nearly every construction here is a small point set `B` copied onto every site
of a unit lattice, and its density has a closed form:

```
edges / n  ->  z/2 + u(B) / |B|
```

That single law orders the whole family — the prism of unit triangles at `4n`,
each up/down doubling adding `0.5`, centered hexagons at `4.71n` — and it says
what to feed it: not another lattice, but the densest small unit-distance set
you have. **A fixed basis only ever moves the constant**; escaping `Θ(n)`
requires the basis to grow with `n`, which is what the rotated flower sums do at
`Θ(n log n)`.

Separately, Erdős' rescaling trick — find the most popular pairwise distance and
scale it to 1 — turns out to do nothing for flower sums and a great deal for the
triangular lattice, where it beats Erdős' own square-grid version at every size.

![the Eisenstein grid](./docs/eisenstein.png)

That is the strongest construction here, and it is just the triangular lattice
with the scale chosen so its fattest distance shell becomes the unit. The points
never move; only which pairs count changes.

[All of it, with the arithmetic](./docs/constructions.md) ·
[growth rates and bounds](./docs/growth.md)

## Documents

| | |
|---|---|
| [method.md](./docs/method.md) | how the solver works, why the naive version fails, auditing, the live view |
| [constructions.md](./docs/constructions.md) | the `B ⊕ lattice` law, the prism, flower sums, the rescaling trick, and why these motifs emerge from the solver unaided |
| [growth.md](./docs/growth.md) | big-O for every family, the standing bounds, measured slopes, runtime cost |

## Layout

| file | what it holds |
|---|---|
| `springs.py` | the four spring laws and the failure threshold |
| `energy.py` | total energy and analytic gradient, blocked over pairs |
| `sim.py` | annealing schedule, integrators, detached-ball rescue |
| `polish.py` | Levenberg–Marquardt projection onto exact unit distances |
| `counting.py` | promoting springs to edges, and the overlap audit |
| `solver.py` | the anneal → project → reheat pipeline and restarts |
| `init.py` | starting configurations, `B ⊕ lattice`, the rescaling trick |
| `known.py` | proven optima, baselines, the `r-min` default |
| `render.py` | live 2D view and static PNG output |
| `cli.py` | argument parsing and the three run modes |

`scripts/` holds the figure generator plus four exploration scripts:
`explore_hex.py`, `explore_prism.py`, `explore_rescale.py` and `asymptotics.py`.

Available starts: `random`, `triangular`, `square`, `erdos_grid`,
`eisenstein_grid`, `hex_flower`, `hex_minkowski`, `prism`, `prism_double`,
`hex_lattice`, `moser`, `circle`.

The solver needs only **numpy**; matplotlib is used for the view and PNGs.
Gradients are analytic and finite-difference tested for every law, with and
without failure and core repulsion active. `pytest tests -q` runs 158 tests.

## Caveats

- `u(n)` for `n ≤ 14` is quoted from the literature (Schade, 1993); cross-check
  against OEIS A186705 before relying on it. Nothing else in the code depends on
  those numbers — they are only used for scoring.
- The density law assumes the basis sits at a generic rotation to the lattice.
  Special angles can do better *or* collapse into duplicate points; `--init`
  uses tested generic values, and the audit catches the rest.
- The projection is a dense `2n × 2n` solve below `--n 250` and matrix-free
  conjugate gradients above it. Both paths are tested to agree; the CG path is
  the one to watch if large runs look wrong.
- Energy and gradient are `O(n²)` per step by construction — it is a complete
  graph. Blocking bounds the memory, not the work.
- The audit's notion of "distinct" is a tolerance (`1e-6`), not exact
  inequality, which is the conservative choice.
