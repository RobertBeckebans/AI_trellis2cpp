# Progress — tangent-space-normal-bake, phase 1: source normals from the projection

- **Issue:** none — tracked by plan key
- **Plan:** [`docs/plan/tangent-space-normal-bake.md`](../plan/tangent-space-normal-bake.md) — Phase 1
- **Branch:** `tangent-space-normal-bake` / optional
- **Date:** 2026-08-13
- **Commits:** <to be added after review>

## Goal of the phase

Get the dense source mesh's shading normal out of the closest-surface
projection, so a later phase can bake a tangent-space normal map without
paying for a second BVH traversal. Nothing user-visible changes yet.

## What was done

**`t2print::project_pbr_and_normals`** beside the existing pair. It takes a
`source_normals` array (3 floats per source vertex, the shape
`t2glb::PreparedMesh::normals` already has) and fills an `out_normals` array of
three floats per query.

**One dispatch instead of three.** The backend selection, input validation and
error handling moved into an internal `project_dispatch`, and
`project_pbr` / `project_pbr_backend` / `project_pbr_and_normals` are now thin
forwarders onto it. The old `project_pbr_backend` body *was* that dispatch, so
this is a move rather than a rewrite, and the CGAL/tinybvh branch structure is
unchanged.

**The normal rides on the existing hit.** `project_with` already had the
triangle and the clamped barycentric weights in hand for the six PBR channels;
the normal is three more interpolated channels behind an optional pointer. No
second traversal, and the material path is untouched when no normals are asked
for.

**Renormalization, and an explicit "no sample".** A barycentric blend of unit
vectors is not unit, so the result is renormalized. Where the three vertex
normals cancel, the output is `(0,0,0)` — the header states that callers must
read that as "no direction" rather than as a direction. `prepare_faces` rejects
a `source_normals` array that does not cover every source vertex instead of
indexing past its end.

## Verification

New section in `tests/test_print_remesh.cpp`, built on a 96×48 UV sphere whose
exact normal at every vertex is the unit position, queried from a shell at
r = 1.3 so the closest surface point is essentially radial and the ground truth
is closed-form. Measured on the ROCm/HIP Release build (backend default
`tinybvh`, CGAL present so both ran):

| Check | Result |
|---|---|
| tinybvh normal vs analytic | **0.6009 deg** worst over 4096 samples |
| CGAL normal vs analytic | **0.6009 deg** worst |
| Unit length | worst `|len-1|` = **4.27e-08** |
| PBR channels with vs without normals | **bit-identical** |
| tinybvh vs CGAL normal | worst **0.02324 deg** |
| Truncated `source_normals` | rejected |

The 0.6 deg residual is the tessellation's own budget (a 96×48 sphere has
~3.75 deg facets and the interpolated normal stays inside that cone), and the
two backends reproduce it to four digits — they share the bias exactly and
differ only by rounding.

`ctest --test-dir build -C Release -LE model`: 8/8 passed (`preprocess` skips
for missing fixtures, as before this change).

## Findings

**The existing backend-comparison comment overstates its case.** It says the
only legitimate difference between tinybvh and CGAL is "which triangle wins a
near-tie at a shared edge". Counting rather than just maxing shows otherwise:
**2021 of 4096** normal samples differ at all, i.e. roughly half, though by at
most 0.023 deg. The cause is not ties — CGAL constructs the closest point with
exact predicates in double while tinybvh works in float, so the barycentric
weights differ by rounding across the entire field. The material comparison has
always seen the same effect; it only ever inspected the maximum, which is why it
reads as a tie there.

Consequence for the test design: a count-based assertion would carry no signal.
The normal test therefore asserts that **both** backends hit the analytic normal
within the same budget — a wrong-surface pick on a sphere misses by degrees,
which no rounding argument absorbs — and reports the pairwise difference as a
measurement rather than gating on it. The misleading claim in the material
section is corrected in place.

## Not done in this phase

- No caller uses the new entry point yet; `mesh_export` is Phase 3.
- No C-API surface, so **no `T2_CAPI_ABI_VERSION` bump** here. The bump lands
  in Phase 4 with the `t2_bake_projected_glb` flag.

## Formatting

`format_code.bat` was **not** run: this machine has clang-format 21.1.8 while
the project mandates 18.1.8, so running it would reformat unrelated code. New
code was written to match the surrounding style by hand; a reviewer with the
pinned version should run the formatter before committing.
