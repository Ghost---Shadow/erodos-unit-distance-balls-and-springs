# Constructions: one law covers all of them

*Part of the [Erdős unit distance](../README.md) project.*

Every construction below is a small point set `B` copied onto every
site of a unit lattice — a Minkowski sum `B ⊕ lattice`. Counting the edges per
site gives `z/2 · |B|` inside each basis point's own lattice copy, plus one edge
for every unit pair *within* `B`. So the density has a closed form:

```
edges / n  ->  z/2 + u(B) / |B|
```

Both terms are levers.

**The base lattice sets the constant**, and it is worth being explicit about
which lattice that is, because it goes by three names.

![why the hexagonal lattice is the 60-degree rhombus lattice](./rhombus.png)

Take two unit vectors 60° apart and tile the plane with the cell they span.
That cell is a **rhombus** with four unit sides. Its long diagonal is √3, but
its *short* diagonal — the one joining the two 60° corners — is also exactly 1,
because those three points form an equilateral triangle. So the short diagonal
cuts the rhombus into two unit **triangles**, which is why the same point set is
called the triangular lattice. And every point ends up with six neighbours at
distance 1, arranged in a **hexagon**, which is why it is also called the
hexagonal lattice. One point set, three names, depending on whether you look at
the cell, the pieces it cuts into, or the neighbourhood of a point.

Counting edges per point from any of those views agrees: six neighbours shared
between two points each gives `6/2 = 3`. From the cell view, each rhombus
contributes two of its four sides (the other two belong to neighbouring cells)
plus its short diagonal, which belongs to no one else — again 3.

A square cell has the same two-sides-per-cell, but *neither* diagonal is unit
(both are √2), so there is nothing to add: `4/2 = 2`. Hence the rhombic lattice
carries exactly **one more unit distance per point**, and that gap of 1.00n
persists across every basis, which is the two rows of the figure below.

One naming trap: "hexagonal lattice" here means this triangular point set, not
the *honeycomb* (graphene) pattern where each point has only three neighbours.
The honeycomb is not a lattice at all — it is this lattice with a two-point
basis — and it is much worse, at 1.33n.

**The basis sets the rest**, and says what to feed it: not another lattice, but
the densest small unit-distance set you have.

| basis `B` | `\|B\|` | `u(B)` | on rhombic | on square | what it is |
|---|---|---|---|---|---|
| one point | 1 | 0 | 3n | 2n | the plain lattice |
| unit segment | 2 | 1 | 3.5n | 2.5n | two sheets joined by unit rungs |
| unit triangle | 3 | 3 | **4n** | 3n | the prism, stacked and tiled |
| two triangles a unit apart | 6 | 9 | **4.5n** | 3.5n | the prism doubled |
| centered hexagon | 7 | 12 | **4.71n** | 3.71n | wheel `W₆` on every site |
| flower sum, k=2 | 49 | 168 | **6.43n** | 5.43n | see below |

![construction family](./constructions.png)

Columns 1–4 of that figure are the basis products above. Column 5 is the same
base lattice under Erdős' rescaling trick instead — not a basis product at all,
but the natural comparison, since it is the other way of getting more unit
distances out of the same lattice. It wins on both rows (4.45n against 3.49n on
the rhombic base, 3.28n against 2.83n on the square), which is the subject of
the next section.

Every entry is verified numerically and audited for overlapping points; the
measured densities sit just below their limits because a finite patch has a
boundary. `scripts/explore_prism.py` reproduces the table.

**The prism.** Triangle ABC joined to DEF with A–D, B–E, C–F all unit. In the
plane that *forces* DEF to be a translate of ABC by a unit vector — there is no
other realisation — so tiling means translating by two unit vectors `v` and `w`.
Setting the angle between them to exactly 60° makes `v − w` a unit vector too,
which connects the diagonal neighbours as well and lifts the count from 3n to

```
3[ab + (a-1)b + a(b-1) + (a-1)(b-1)]  ->  4n
```

verified exactly. The triangle's rotation relative to `v` and `w` changes
nothing about the count, so the default picks a rotation with comfortable point
separation (0.52) rather than one that merely avoids collisions.

**Doubling up and down** adds a `{0, u}` factor to the basis, which adds exactly
one unit pair per two basis points: **+0.5 density per doubling**, measured at
3.75 → 4.25 → 4.75 → 5.25n for zero through three doublings.

**A prism of triangular lattices** — two lattice sheets joined by unit rungs —
is the `|B| = 2` row, and is therefore *worse* than the prism of triangles. A
segment carries only half a unit distance per point; a triangle carries one.
The lattice you stack is the *base*, not the basis; stacking lattices on
lattices adds nothing that raising `u(B)/|B|` would not add more of.

**The centered hexagon** (a hub plus six unit vectors, the wheel `W₆`) has 12
unit distances on 7 points, which is exactly `u(7)`, and it is the first thing
the solver rediscovers. Note that *tiling* it gives nothing new — its six
vectors generate the triangular lattice, so a honeycomb of centered hexagons
**is** the triangular lattice. The true honeycomb (graphene, 3 neighbours each)
is far worse, at 1.33n.

**Flower sums.** What does work on the hexagon is Minkowski-summing **rotated**
copies of it. Summing `k` flowers at generic angles gives `7^k` distinct points
carrying exactly `12·k·7^(k-1)` unit distances — that is `(12/7)·log₇(n)·n`, so
it grows like `n log n`. Verified exactly for `k = 1..4`, up to 2401 points and
16464 unit distances, all passing the audit. Since `u(B)/|B|` for a flower sum
itself grows logarithmically, feeding one in as the lattice basis inherits that
growth.

### The Erdős rescaling trick, and where it applies

Erdős' grid is really two independent steps, and separating them is useful:

1. build a point set whose pairwise **squared distances are integers**;
2. find the value realised by the **most** pairs and scale by `1/sqrt` of it,
   turning all of those pairs into unit distances.

Step 2 is generic — `init.rescale_to_popular` applies it to any point set, and
since it is only a similarity it can never merge points, so a valid
configuration stays valid. Step 1 is the arithmetic part, and it is what
actually does the work: integers have wildly uneven numbers of representations
as a sum of two squares, so the distance histogram has a tall spike to aim at.

**So can you rescale a flower sum? No — and the reason is the interesting
part.** The two tricks pull in opposite directions. Rescaling needs a
*concentrated* distance spectrum with one huge spike. Minkowski sums with
generic rotations work by *spreading* the spectrum thin so that the unit
distance is the only spike. Measured at n = 900:

| construction | distinct distances | rescaling gain |
|---|---|---|
| triangular lattice | 273 | **2.59×** |
| square lattice | 372 | 2.91× |
| prism of triangles | 688 | 1.32× |
| centered hexagon × lattice | 1498 | 1.02× |
| rotated flower sums | 8260 | **1.00×** |

The flower sums realise 8260 distinct distances across 404550 pairs — 49 pairs
per distance on average — and their tallest spike is already the unit distance.
There is nothing more popular to rescale to. The more a construction fragments
its spectrum to win unit distances directly, the less the rescaling trick has
left to offer it.

**But applying step 2 to the triangular lattice pays handsomely.** Erdős used
the square lattice, where squared distances are `i² + j²`. On the triangular
lattice they are `i² + ij + j²` instead — norms of Eisenstein integers, the
Loeschian numbers — whose representation counts spike on integers built from
primes ≡ 1 mod 3. The spike is taller, and the result beats the square-grid
version at every size measured:

| n | 100 | 300 | 900 | 2700 |
|---|---|---|---|---|
| Erdős grid (square) | 288 | 1212 | 4944 | 19568 |
| Eisenstein grid (triangular) | **411** | **1470** | **6706** | **25225** |

![the Eisenstein grid](./eisenstein.png)

Left: the distance spectrum of the triangular lattice at `n = 900`. The winner
is `r = 91 = 7 × 13` — both factors ≡ 1 mod 3, and every runner-up is the same
kind of number (`133 = 7 × 19`, `49 = 7²`, `217 = 7 × 31`). That is the whole
mechanism: primes ≡ 1 mod 3 split in the Eisenstein integers, and each one
multiplies the number of ways `r` can be written as `i² + ij + j²`.

Middle: the resulting point set. It *is* the triangular lattice — the points
have not moved, only the scale — but far more pairs now sit at distance 1.

Right: why. The central ball's unit neighbours are the `r = 7` shell, twelve of
them, reaching past its six nearest. Rescaling does not add points or edges to
the lattice; it re-chooses which shell counts as unit, and picks the fattest one.

That is `--init eisenstein_grid`, and it is the strongest construction here.
Its most popular distance is `sqrt(7)`, and 7 ≡ 1 mod 3 exactly as the theory
predicts. `scripts/explore_rescale.py` reproduces all of the above.

Available as `--init prism`, `prism_double`, `hex_lattice`, `hex_minkowski`;
`init.lattice_product(n, basis, lattice=...)` builds any other combination,
and `init.density_limit(basis, lattice)` predicts what it will score.

## The lattices are not assumed, they emerge

Worth saying plainly, because it is easy to read this repo backwards: the
solver is never told about any of the constructions above. It starts from
uniform random points and only ever minimises the spring energy. The
structures show up on their own.

The clearest case is `n = 7`. The solver converges to the centered hexagon —
hub plus six unit vectors, 12 unit distances — which is both the proven
optimum `u(7)` and the building block that the flower sums are made of. It is
the top-left panel of the gallery in the [README](../README.md), and nothing in the
code knew it was a target. Run `python run.py --n 7` and watch it find it.

At `n = 20` to `30` the outputs are visibly lattice fragments: rows of unit
triangles, hexagonal neighbourhoods, and translated copies of small motifs.
That is the whole basis of the `B ⊕ lattice` family, arrived at from below.

This makes for a usable workflow, and it is how several things here were found:

1. run the optimiser at a size it handles well (`n ≲ 30`);
2. look at what it converged to, and name the motif by eye;
3. generalise it analytically — tile it, Minkowski-sum it, rescale it;
4. verify the closed form numerically and audit it for coincident points.

Step 2 is human inspection and there is currently no substitute for it in this
code: the solver finds structures far more readily than it explains them. Step
4 matters as much as step 3, since a plausible-looking generalisation can
easily be a duplicate-point artefact rather than a construction — which is what
the audit exists to catch.

The honest limit is that this only works where the solver is strong. Past about
`n = 30` it stops finding anything the constructions do not already beat, so
the pipeline runs out of new motifs exactly where you would most want them.
