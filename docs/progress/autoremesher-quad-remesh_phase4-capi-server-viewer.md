# Progress — autoremesher-quad-remesh, phase 4: C-ABI, server, viewer

- **Issue:** none — tracked by plan key
- **Plan:** [`docs/plan/autoremesher-quad-remesh.md`](../plan/autoremesher-quad-remesh.md) — Phase 4
- **Date:** 2026-08-12
- **Commits:** <to be added after review>

## Goal of the phase

Make the quad stage reachable from the Go demo server and its viewer, which is
the first point where the whole plan becomes usable rather than testable.

## What was done

**C-ABI.** `t2_quad_remesh_available()`, `t2_prepare_quad_mesh()` and
`t2_quad_mesh_stats()`, documented in the header's existing style.
`T2_CAPI_ABI_VERSION` **11 → 12**, propagated to `const abiVersion` in
`server/engine.go`. The stats ride on `t2_mesh_result`, which is opaque to
hosts, so growing it costs them nothing.

**Go engine.** `HasQuadRemesh()`, `PrepareQuadMesh()` and a `QuadStats` type.

**Fixed a stale gate.** `BakeProjectedGLB` refused to run unless
`HasPrintRemesh()`, i.e. unless CGAL was present — an assumption Phase 2
invalidated when the closest-surface projection moved to tinybvh. The Go layer
was still rejecting a call the C function now handles fine.

**Server.** `quad`, `quads` and `adaptivity` job parameters; the quad stage runs
after the optional wrap in `preparedExportMesh`; the prepare/GLB cache keys
include the quad parameters so switching settings does not serve a stale mesh;
`/api/info` reports `quad_remesh` next to `print_remesh`.

**Quality is reported, not implied.** `/api/export-preview` returns
`X-Quad-Quads`, `-Triangles`, `-Ngons`, `-Boundary-Edges` and `-Area-Retained`
headers, and the viewer renders them under the new control. The export hint
distinguishes all four combinations, and says plainly that wrap + quad is *not*
printable because the quad stage can reopen boundaries.

**Viewer.** A `quad retopology (mid-poly)` checkbox with target-quads and
adaptivity sliders, disabled with a reason when the backend is absent. No CDN,
no build step — same inline pattern as the rest of the page.

## Affected files

- `trellis2_capi.{h,cpp}` — three entry points, ABI 12
- `server/engine.go` — bindings, `PrepareQuadMesh`, `QuadStats`, ABI 12, stale-gate fix
- `server/main.go` — job options, cache keys, caps flag, quality headers
- `server/web/index.html` — control, sliders, stats readout, export hints

## Verification

| Check | Result |
|---|---|
| C++ build, no CGAL, vcpkg manifest | success |
| `ctest -LE model` | **100 % passed**, 7/7 |
| `go build ./...` | success |
| `./format_code.sh` | clean |
| Library loaded by the server binary | ABI check passed at 12, all three new symbols resolved |
| `/api/info` | `quad_remesh: true`, `print_remesh: false` |
| CLI regression | `mesh2glb --quad` unchanged: 532 tris, 243 quads, closed, 98.4 % area |

`go vet` reports four `possible misuse of unsafe.Pointer` warnings, all in the
pre-existing purego helpers (`copyFloats`, `copyInts`, the two bake functions),
none in code added here.

### End to end through the running server

A real generation (coarse 64³ path, synthetic input image), exported twice:

| Export | Result |
|---|---|
| `?components=2` | 110 690 verts, 221 668 tris |
| `?components=2&quad=1&quads=3000` | 3 593 verts, **5 748 tris** |

Reported quality headers for that run:

```
X-Quad-Quads: 2104      X-Quad-Triangles: 347   X-Quad-Ngons: 695
X-Quad-Boundary-Edges: 796
X-Quad-Area-Retained: 0.8717
```

### The number that matters

**On real pipeline geometry the result is open and lossy.** 796 boundary edges
and 87.2 % retained area, against 0 boundary edges and 98.4 % on every synthetic
sphere fixture so far. Only 2104 of 3146 faces are quads (67 %), with 695 n-gons.

That is the first honest look at this stage on TRELLIS output, and it says the
synthetic fixtures were far too kind. It does not invalidate the feature — a
mid-poly mesh with a projected PBR atlas is still what the plan set out to build
— but it does mean the UI must keep showing these numbers rather than implying a
clean retopology, which is why the headers and the readout exist.

Caveat: this was the **coarse 64³ marching-cubes** path, not the 512³ dual grid.
The dual grid is denser and differently non-manifold, so these numbers are a
first data point, not the answer.

## Open points

- The 512³/1024³ path is still unmeasured; the coarse path was used because it
  runs in seconds on the CPU build.
- 796 open edges deserves investigation before this is offered as a default:
  is it the component filter, the input preparation, or AutoRemesher's quad
  extraction? The `--quad`/`min_area_retained` knobs are untuned against it.
- The `ProgressFn` in `t2quad::remesh` is still not wired to the server's
  progress callback, so a long remesh looks like a stall in the UI. On the
  coarse mesh it was fast enough not to matter; on 512³ it likely will.
- No test covers the new C-ABI entry points (Phase 5).
- Two persisted jobs from this verification are in `generations/` — user data,
  left in place.

## Next phase

Phase 5: tests for `quad_remesh` (determinism, quad-ratio floor, index-range
invariants, and an explicit test that boundary edges may exist), the benchmark
table, and the documentation pass.

## Proposed commit message

```text
Expose quad remeshing through the C ABI, server and viewer
```
