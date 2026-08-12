# Progress — autoremesher-quad-remesh, phase 3: the quad_remesh module

- **Issue:** none — tracked by plan key
- **Plan:** [`docs/plan/autoremesher-quad-remesh.md`](../plan/autoremesher-quad-remesh.md) — Phase 3
- **Branch:** `autoremesher-quad-remesh` / optional
- **Date:** 2026-08-12
- **Commits:** <to be added after review>

## Goal of the phase

Turn the vendored core into a library stage with the same encapsulation as
`print_remesh`, give it the input preparation the dual grid needs, and make the
result reachable end to end so the output can actually be looked at.

## What was done

**`quad_remesh.{h,cpp}`** — `t2quad::available()`, `remesh()` and
`triangulate()`, no vendored type in the public header, compiles and reports
unavailable without the backend. Output is a quad-*dominant* face stream
(`out_faces` + `out_face_sizes`) so nothing downstream has to assume quads.

**Input preparation**, in this order: drop zero-area and duplicate faces →
optional `meshopt_simplify` to an input triangle budget → split non-manifold
vertices. The splitter treats two faces as connected at a vertex only through
an edge used by exactly two faces overall, then gives each connected group its
own copy of the vertex. That breaks bow-ties and edges shared by three or more
faces, which is what the halfedge mesh inside AutoRemesher cannot represent.

**A failure policy that does not lie.** AutoRemesher returns success even when
it silently skipped islands, so `remesh()` compares output to input surface area
and refuses below `min_area_retained` (default 0.5) instead of handing back a
half-remeshed model. `QuadRemeshStats` reports quad ratio, boundary edges,
retained area, split vertices and dropped faces.

**`t2glb::prepare_quad_mesh`** mirrors `prepare_print_mesh`: component filter →
quad remesh → triangulation (shorter-diagonal split for quads) → per-vertex PBR
by closest-surface projection.

**`mesh2glb --quad [target_quads]`**, combinable with `--print`. The projected
bake now runs for `print_wrap || quad_remesh`, which is what makes Phase 2's
tinybvh path reachable from a CLI for the first time.

## Affected files

- `quad_remesh.{h,cpp}` — new
- `mesh_export.{h,cpp}` — `quad_remesh_available()`, `prepare_quad_mesh()`, `QuadMeshStats`
- `examples/mesh2glb.cpp` — `--quad`
- `CMakeLists.txt` — module sources, meshoptimizer sources, target-level defines
- `THIRD_PARTY.md` — corrected meshoptimizer attribution

## Deviations / fixed along the way

1. **meshoptimizer was vendored but never compiled.** No project source used it;
   its four `.cpp` files were in the tree but in no target, so `meshopt_simplify`
   failed to link. `quad_remesh.cpp` is its first actual consumer. The
   `THIRD_PARTY.md` entry claiming it was used by `mesh_export.cpp` was wrong and
   has been corrected.
2. `_USE_MATH_DEFINES` / `NOMINMAX` moved from per-source properties to the
   `trellis2` target: `quad_remesh.cpp` includes the AutoRemesher headers too and
   hit the same missing `M_PI`.
3. Three doc comments in `mesh_export.h` ended up above the wrong declaration
   after the insertions and were moved back.

## Verification

| Check | Result |
|---|---|
| Build, no CGAL, vcpkg manifest | success |
| `ctest -LE model` | **100 % passed**, 7/7 |
| `./format_code.sh` | clean |
| `mesh2glb sphere.t2mesh out.glb 512 --quad 2000` | 2304 tris → 532 tris / 266 verts, 243 quads / 14 tris / 10 ngons, **0 boundary edges**, 98.4 % area |
| GLB contents | `baseColorTexture`, `metallicRoughnessTexture`, `TEXCOORD_0` present, no `COLOR_0` — a real UV atlas baked through tinybvh, no CGAL |

### The measurement that corrects Phase 1

Running the defect matrix through the new preparation, with the splitter
toggled:

| Fixture | split off | split on | Phase 1 (no preparation) |
|---|---|---|---|
| sphere | 98.4 % | 98.4 % | 99.2 % |
| two disjoint spheres | 96.9 % | 96.9 % | 101.7 % |
| shared (non-manifold) vertex | **97.9 %** | 97.9 % (1 vertex split) | **49.9 %** |
| duplicated face | 98.3 % | 98.3 % | 99.3 % |
| dangling fin | 98.3 % | 98.3 % (1 split) | 99.5 % |
| all defects combined | **96.7 %** | 96.7 % (2 splits) | **9.1 %** |

The pathological case went from 9.1 % to 96.7 % retained area — but **not
because of the splitter**. With splitting disabled the numbers are identical, so
the fix came from dropping zero-area triangles: the UV-sphere fixtures carry 96
degenerate pole triangles each, and it was the *combination* of those with a
non-manifold vertex that destroyed the result. Removing either one is enough.

This corrects the Phase 1 conclusion, which named the non-manifold junction as
"the killer" on the strength of an isolation run where every fixture also
contained degenerate poles. The splitter is kept because it is cheap, it fires
correctly (1 and 2 junctions detected), and the failure it guards against is
silent — but it is **not measured to be necessary**, and the header comment
claiming otherwise has been corrected.

## Open points

- **Still no real 512³ pipeline mesh.** Every number above is from synthetic
  fixtures. Real dual-grid geometry is the only thing that can decide whether the
  splitter earns its place, what `target_quads` should default to, and whether
  `input_triangle_budget` needs to be on by default.
- `min_area_retained = 0.5` is a guess, not a calibrated threshold.
- `input_triangle_budget` defaults to off and is untested on real data.
- No unit test yet for `quad_remesh` (Phase 5): determinism, quad-ratio floor,
  index-range invariants, and an explicit test that boundary edges may exist.
- The C-API and server still have no quad entry points (Phase 4), so this is
  CLI-only. The stale `HasPrintRemesh()` gate on `BakeProjectedGLB` in
  `server/engine.go` is still there.

## Next phase

Phase 4: `t2_quad_remesh_available()` / `t2_prepare_quad_mesh()`, the
`T2_CAPI_ABI_VERSION` bump, `server/engine.go`, and the viewer toggle — the point
at which this becomes testable in the demo UI.

## Proposed commit message

```text
Add the quad_remesh module and reach it from mesh2glb
```
