# Quad remeshing — the mid-poly topology stage

The quad stage turns the dense dual-grid soup into quad-dominant, mid-poly
topology, backed by the vendored MIT AutoRemesher core
(`third_party/autoremesher/`, wrapped by `src/quad_remesh.{h,cpp}`).

It is **not** a replacement for the CGAL Alpha Wrap print path, and the two
answer different questions:

| | Alpha Wrap (`TRELLIS2_CGAL`) | Quad remesh (`TRELLIS2_AUTOREMESHER`) |
|---|---|---|
| Guarantees | closed, 2-manifold, printable | none — may leave boundary edges |
| Output | dense triangles | quad-dominant, mid-poly |
| Vertices | all new (offset surface) | all new (parameterized) |
| License | GPL, encapsulated | MIT |

They compose: wrapping first hands the remesher the closed 2-manifold it
expects, which is measurably the better input (see below). The pair is still
not guaranteed watertight, so the viewer reports the boundary-edge count rather
than implying a clean retopology.

## Benchmarks

Absolute times are machine-specific — a Windows clang/Release build, CPU-only
stage, CPU model unrecorded. The ratios are the part worth reading.

### Synthetic fixtures (Phase 1)

`--target-quads 20000` for the dense meshes, `2000` otherwise.

| Input | Tris in | Wall | Peak RSS | Faces out | Quads | Boundary edges | Area retained |
|---|---|---|---|---|---|---|---|
| cube | 12 | 0.05 s | 7 MiB | 139 | 100.0 % | 0 (closed) | 94.5 % |
| sphere | 2 304 | 0.27 s | 26 MiB | 295 | 95.6 % | 0 (closed) | 99.2 % |
| dense sphere | 65 536 | 5.09 s | 236 MiB | 1 936 | 98.4 % | 0 (closed) | 99.9 % |
| dense sphere | 262 144 | 8.90 s | 281 MiB | 2 295 | 99.0 % | 0 (closed) | 99.9 % |

Cost grows clearly **sublinearly** in triangle count: 4× the triangles for 1.75×
the time. That is what makes the stage affordable on real geometry at all.

### Real pipeline geometry

| Source | Tris in | Wall | Faces out | Quads | Boundary edges | Area retained |
|---|---|---|---|---|---|---|
| coarse 64³ dual grid, raw | 221 668 | — | 3 146 | 66.9 % | **796** | 87.2 % |
| 1024³ wrap, target 20 000, adaptivity 1.0 | 369 124 | 72.4 s | 8 004 | 89.8 % | 5 | 92.0 % |
| 1024³ wrap, target 50 500, adaptivity 0.5 | 369 124 | 120.3 s | 26 025 | 96.7 % | 0 | 95.5 % |
| 1024³ wrap **after boundary capping**, target 20 000 | 102 946 | — | 6 796 | 93.3 % | 6 | 94.7 % |

Three things this table says:

1. **Wrap first.** On one 1024³ model, remeshing the raw dual grid gave 9 030
   boundary edges at 72.4 % area retained; remeshing the *wrap* of the same
   model gave 5 edges at 92.0 %. AutoRemesher wants a closed 2-manifold, and
   the dual grid is neither. (The coarse 64³ row above is the raw path on a
   different, much smaller model, hence its milder 796 / 87.2 %.)
2. **Adding the quad stage after the wrap pays for itself.** The projected bake
   that follows then unwraps ~16 k triangles instead of the wrap's 369 k, which
   is the single most expensive part of the export.
3. **Capping the wrap source matters** (`t2print::alpha_wrap`). Before it, the
   wrap returned a two-walled membrane, so the remesher faithfully reproduced
   an interior shell — same quality figures, twice the geometry, and a model
   whose enclosed volume was near zero.

### `--target-quads` is a hint, not a count

20 000 requested produced ~2 300 faces on the dense spheres. The value maps to
`targetTriangleCount = quads * 2` the way the AutoRemesher UI does, feeds a
voxel size, and adaptivity then moves the result a long way from it. Nothing in
the UI or the API may present it as a face count.

## What breaks it

Phase 1 isolated the failure modes on synthetic defects:

| Input defect | Area retained |
|---|---|
| two disjoint components | 101.7 % — fine |
| duplicated face | 99.3 % — fine |
| dangling triangle fin | 99.5 % — fine |
| **one shared (non-manifold) vertex** | **49.9 %** — one of two spheres survives |

The killer is the non-manifold *junction*, and the failure is **silent**:
`remesh()` returns success. Two consequences are built into the wrapper —
`split_non_manifold` breaks such junctions open before parameterization, and
`min_area_retained` turns a silent drop into an error, since the API itself
reports nothing about islands it skipped.

## Tests

`tests/test_quad_remesh.cpp` (ctest `quad_remesh`, no model assets; skips with
77 in a build without the backend) asserts the structural invariants rather
than the output, which is free to move: face-size partitioning, index range and
finiteness, no repeated corner within a face, statistics matching the mesh they
describe, a 50 % quad-ratio floor, `triangulate()` covering every face exactly
once, and two runs on one input being byte-identical. It deliberately does
**not** assert closedness — boundary edges are a documented legitimate outcome,
and the test says so.
