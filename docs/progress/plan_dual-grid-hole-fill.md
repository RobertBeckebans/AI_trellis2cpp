# Progress — plan: the dual-grid hole fill was not resolution-normalised

- **Issue:** none — plan key `plan` ([`docs/PLAN.md`](../PLAN.md), mesh
  extraction), because this is a pre-existing defect rather than a planned phase
- **Branch:** `resolution-1536`
- **Date:** 2026-08-14
- **Commits:** <to be added after review>

## Goal

Small holes had been showing up in generated meshes, seed-dependent and more
frequent the higher the tier. They were found while chasing the 1536 cascade,
but they **predate it** — the same holes occur at 512³ and 1024³ — so this is
its own defect and its own change.

## Symptom

Scattered small openings over the surface. Confirmed as real boundaries in
wireframe, not a shading artefact. The same object and seed could come out
clean at 512³ and holed at 1024³.

## Cause

`fdg::fill_holes` (`examples/flexible_dual_grid.h`) closes each boundary loop up
to `max_loop = 64`, and that limit counts **boundary edges**:

```cpp
inline void fill_holes( Mesh& m, int max_loop = 64, int max_passes = 4 )
// Each loop up to `max_loop` vertices is fan-triangulated from its first
// vertex. Large loops (genuine openings) are left unfilled.
```

An edge count is not normalised against the grid. The same *physical* opening
has proportionally more boundary edges on a finer grid:

| tier | boundary edges of a same-sized hole | vs. limit 64 |
|---|---|---|
| 512³ | ~50 | filled |
| 1024³ | ~100 | **left open** |
| 1536³ | ~150 | **left open** |

So an opening that gets closed at one tier simply stops being closed at the
next. That accounts for both observations: holes increasing with resolution,
and the seed-dependence — whether a given loop lands above or below 64 is an
accident of the geometry.

`drop_small_components` beside it uses `min_frac` relative to the total face
count and is already scale-invariant. This one was not.

## What was done

`src/trellis2_capi.cpp`: the limit scales with the grid, putting the threshold
back at a fixed physical size.

```cpp
fdg::fill_holes( mesh, std::max( 64, 64 * grid / 512 ) );
```

64 at 512³ (unchanged), 128 at 1024³, 192 at 1536³.

The cap is not removed, and the reason is in the comment: a loop is
fan-triangulated from its first vertex, which degrades on large non-planar
boundaries. This raises the size of opening that gets a rough fill; it does not
make filling free.

Alongside it, the final mesh is now measured under `TRELLIS2_TIMING`:

```text
[mesh] grid 1536: … verts, … tris, N boundary edges (X% of E), M non-manifold edges, fill limit 192
```

A boundary edge belongs to exactly one triangle, so a closed surface has none.
This exists because a wireframe screenshot at three million triangles cannot
separate "the limit is still too low" from "the extractor left genuine
openings" — and those need opposite fixes.

## Affected files

- `src/trellis2_capi.cpp` — grid-scaled `fill_holes` limit, boundary/non-manifold
  edge report

## Verification

Reported by the reviewer on the reported cases, on the card:

| case | before | after |
|---|---|---|
| jester, 1024³ | holes | **none** |
| armchair, 1024³, seed 2026 | holes | **none** |
| armchair, 1024³, seed 42 | holes | fewer |
| armchair, 512³ | no hole-producing seed found | unchanged (limit is 64 either way) |
| jester, 1536³ | holes | **unchanged** |

`ctest --test-dir build-… -C Release -LE model` → 9/9 (3 of those are SKIPs for
missing assets/config, unrelated to this change).

Not verified numerically: there is no automated gate for this. See below.

## Open points

- **1536³ is unchanged.** Either the loops there exceed even the scaled limit,
  or they are not fillable loops at all — the extractor left genuine openings,
  or `max_passes = 4` stops before convergence. Those need opposite fixes, and
  the boundary-edge report added here is what will tell them apart: run 1024
  and 1536 side by side and compare the counts.
- **`fdg::extract` has no manifoldness test.** `docs/VERIFICATION.md` credits
  `test_marching_cubes` for "dual-grid mesh extraction", but that test exercises
  `examples/marching_cubes.h`, a different extractor. An analytic fixture
  through `fdg::extract` + `drop_small_components` + `fill_holes`, gated on
  boundary edges and Euler characteristic, would run on the CPU with no assets
  — and would have caught this. That is the gap worth closing next.

## Proposed commit message

```text
Scale the dual-grid hole fill with the grid, so finer tiers stop losing to it

fill_holes caps the boundary loops it closes at 64 counted in EDGES, which is
not normalised against the grid: the same physical opening has more boundary
edges at a finer grid and stops being filled somewhere between tiers. Measured:
holes gone at 1024, unchanged at 1536. Boundary- and non-manifold-edge counts
are now reported under TRELLIS2_TIMING to tell the remaining cases apart.

Predates the 1536 tier - the same holes occur at 512 and 1024.

Detail in docs/progress/plan_dual-grid-hole-fill.md.
```
