# Method: springs, annealing, and exact projection

*Part of the [Erdős unit distance](../README.md) project.*

## Why a plain spring graph does not work

Two more things had to be right, both found by the numbers coming out wrong.

**Far springs must let go.** With ordinary Hooke springs a pair at distance 5
pulls five times harder than a pair at 1.2, so far pairs dominate the gradient
and crush everything into a blob. Every spring law here except `hooke`
*saturates*: past a strain scale `σ` the force dies away and the spring
effectively detaches. Annealing `σ` from wide to narrow interpolates from a true
complete Hooke graph down to a network where only near-unit springs still pull.

**The well depth must not depend on `σ`.** The obvious normalisation — fix the
small-strain stiffness, let the energy saturate at `k σ²` — makes the whole
objective *evaporate* as `σ` anneals to zero, and the assembly diffuses apart
exactly when it should be locking in. Instead the depth is fixed at one unit of
reward per satisfied spring, so

```
energy = depth × (number of springs not satisfied)
```

and minimising it *is* maximising the unit-distance count. The stiffness
`depth/σ²` then grows by orders of magnitude during the anneal, which the MD
integrator absorbs by scaling its timestep with `σ` and its damping with `1/σ`
— the dynamics stay self-similar and simply tighten.

## How a run works

![one run, start to finish](./anneal.png)

Each round has two halves, because the relaxation is good at combinatorics and
bad at precision — it will happily park a spring at 1.0004.

1. **Anneal** the complete spring graph under damped molecular dynamics with
   Langevin kicks, narrowing `σ`. This decides *which* pairs want to be unit.
2. **Project**: freeze that edge set and solve the geometry exactly with
   Levenberg–Marquardt, driving residuals to ~1e-15. Then look for near-misses
   the tightened geometry pulled into range, add them, and re-solve.

The drop in that last panel is the point of step 2. Springs "within 0.02 of
unit" are wishful thinking — many of them cannot all be satisfied at once, and
the exact solve prunes the ones that were never realisable. What survives is
geometry, not rounding.

Then **reheat** — restart the anneal from the exact configuration with a smaller
`σ0` — and repeat. This basin hopping is where most of the quality comes from:
sweeps showed extra annealing steps stop paying off past ~2500, because what
limits a run is which basin it fell into, not how carefully it descended.

## Spring failure (optional, off by default)

Independently of the smooth saturation, a spring can be made to simply **snap**:

```bash
python run.py --n 24 --break-lo 0.5 --break-hi 2.0    # squeezed or stretched too far
python run.py --n 24                                  # the default: no failure
```

Squeezed below `break-lo` or stretched past `break-hi`, a spring exerts exactly
zero force. Its energy plateaus at the value it had when it broke, so there is
no force step and no spurious well at the threshold for balls to get caught in.
It composes with the smooth laws and is what makes `--law hooke` usable at all.

It is **off by default**, because the saturating laws already fade a violated
spring out smoothly and the hard cutoff is a second, blunter mechanism on top.
It is not free to leave off, though: with failure at 0.5/2.0 the solver reaches
`u(7) = 12` from ten restarts, and without it `n = 7` needs about 24 restarts to
find the same configuration. Every other `n ≤ 14` is unaffected.

The core repulsion is deliberately *not* subject to failure — it is all that
stops two balls that have snapped every spring between them from merging.

**Stranded balls.** Either way, a ball can end up with nothing close enough to
pull on it: past `break-hi` if failure is on, or simply past a few `σ` of strain
if it is off, where the saturating force is numerically zero. Such a ball is
inert and wastes a point. Every `--rescue-every` steps those get put back into
the assembly, worth a lot at moderate `n` (at `n = 24`, 48 → 57 unit distances).
The reach that defines "stranded" is pinned to `σ0` rather than the current `σ`;
letting it shrink with the anneal makes it approach 1, at which point the rescue
starts relocating balls that are still doing useful work.

## Counts are audited, not trusted

The cheapest way to fake a high score is to stop using `n` distinct points.
Stack 12 balls onto 3 spots a unit apart and every cross-cluster pair "is" a
unit distance: 48 of them, against a true optimum of 27. It is not a point set
at all, but nothing in the energy forbids it, and **the optimiser finds it
immediately** if you let it.

So every reported configuration is re-checked from its coordinates alone, by
code that has no idea how it was produced:

```
audit FAIL: 48 unit distances among 3/12 distinct points, min separation 0
  - 9 ball(s) coincide within 1e-06; the count is inflated by duplicate points
  - count drops from 48 to 3 once coincident balls are merged
```

The audit verifies that no two balls coincide, that merging any coincident ones
would not change the count, that recounting the coordinates reproduces the
claimed edge list, and that every claimed edge really is within `ε` of unit
length and listed once. `--headless` prints it, the benchmark refuses to pass
without it, the live HUD warns on screen, and `SolveResult.valid` carries it.
A restart whose audit fails always loses to one whose audit passes, no matter
how many "edges" it claims.

Three defences keep it passing in the first place: a hard core during the
relaxation, stiffness-scaled so it cannot be overwhelmed as the well narrows;
the same core as a barrier inside the exact solver, which otherwise happily
drives points into exact coincidence; and `--r-min` defaulting to a value
derived from the best construction known at that `n`, so it never excludes a
real answer.

## The live view

![live view](./live_view.png)

<kbd>space</kbd> pause · <kbd>r</kbd> restart · <kbd>n</kbd> new start ·
<kbd>p</kbd> project onto exact unit distances · <kbd>[</kbd> <kbd>]</kbd> change `ε` ·
<kbd>a</kbd> toggle faint springs · <kbd>s</kbd> save PNG + JSON · <kbd>q</kbd> quit

Three layers are drawn. Faint warm/cool lines are springs still negotiating —
warm where compressed, cool where stretched. Dim green are *candidates*: what
the relaxation has agreed on and will hand to the exact solver. Bright green are
the springs within `ε` of unit, the ones actually counted. Watching the faint
set condense into the bright set as `σ` anneals is the method in one picture.

Balls transiently overlapping mid-anneal is normal — the core pushes them back
apart as it converges, and the HUD warns on screen if any are still coincident
at the end.
