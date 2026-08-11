# Progress — autoremesher-quad-remesh, phase 1: vendor the AutoRemesher core

- **Issue:** none — tracked by plan key
- **Plan:** [`docs/plan/autoremesher-quad-remesh.md`](../plan/autoremesher-quad-remesh.md) — Phase 1
- **Branch:** `autoremesher-quad-remesh` / optional
- **Date:** 2026-08-11
- **Commits:** <to be added after review>

## Goal of the phase

Vendor the MIT AutoRemesher 1.0.0 core, make it build inside this project
without Qt and without TBB, and measure what it actually costs and produces —
before any library wiring exists.

## What was done

**Vendoring.** `third_party/autoremesher/` holds the upstream `src/AutoRemesher/`
(28 files), `include/AutoRemesher/` (17 umbrella headers), and
`thirdparty/isotropicremesher/` (11 files), 528 KB total. The upstream directory
layout is preserved because the umbrella headers reach the implementation
through relative paths. Nothing from the Qt application, the shaders, or the
bundled Eigen/TBB copies was imported.

**TBB removed without patching upstream.** The used surface turned out to be
smaller than the plan assumed — only `tbb::blocked_range<size_t>(begin, end)` and
`tbb::parallel_for(range, body)`; `tbb::mutex` and `tbb::combinable` are included
but never used. `t2_tbb_shim.h` implements all four over `std::thread`, and
`compat/tbb/*.h` supply the header names the sources include, so
**the upstream sources carry no TBB patch at all**. The shim handles the two
things that matter here: nested `parallel_for` (AutoRemesher runs parallel loops
inside its per-island parallel loop) is bounded by a global worker budget instead
of multiplying threads, and exceptions from the body are captured and rethrown
after the join instead of crossing a thread boundary into `std::terminate`.

**Logging.** All 35 unconditional `std::cerr` writes now go through `T2_AR_LOG`,
which is silent unless `TRELLIS2_AUTOREMESHER_VERBOSE` is set. Applied as a
replayable `sed`, recorded in `PATCHES.md`.

**Build.** `TRELLIS2_AUTOREMESHER` mirrors the `TRELLIS2_CGAL` probe: on, but
self-disabling with a `message(STATUS ...)` when Eigen 5.x is absent. Eigen comes
from the new `vcpkg.json` manifest; no MPL2 file entered the tree.

**Smoke tool.** `examples/quad_remesh_cli.cpp` — OBJ in, quad OBJ out, reporting
wall time, peak RSS, quad ratio, boundary edges, and surface-area retention.

## Affected files

- `third_party/autoremesher/` — new: vendored core, `LICENSE`, `VERSION.md`,
  `PATCHES.md`, `t2_tbb_shim.h`, `t2_ar_log.{h,cpp}`, `compat/tbb/*.h`
- `vcpkg.json` — new: manifest, Eigen as a dependency, CGAL as a *feature*
- `CMakeLists.txt` — `TRELLIS2_AUTOREMESHER` option, Eigen probe, vendored sources
- `examples/CMakeLists.txt`, `examples/quad_remesh_cli.cpp` — new smoke tool
- `cmake-ninja-win64-rocm-cgal.bat` — `-DVCPKG_MANIFEST_FEATURES=cgal`
- `docs/plan/autoremesher-quad-remesh.md` — phase checked off, Phase 3 corrected

## Deviations / fixed along the way

1. **`VERSION` shadowed `<version>`.** A file named `VERSION` in a directory on
   the include path is found by `#include <version>` on a case-insensitive
   filesystem, which broke every vendored translation unit with errors inside
   "version". Renamed to `VERSION.md`, and the vendor root was dropped from the
   include path since nothing needed it.
2. **`_USE_MATH_DEFINES` / `NOMINMAX`** are set globally by upstream's
   `autoremesher.pro`; without them `M_PI` is undefined. Added as per-source
   compile definitions rather than as a source patch.
3. **`AccelerateSupport` needed no patch** — it is already behind
   `#if AUTO_REMESHER_USE_ACCELERATE`, which we never define.
4. **The smoke tool is standalone.** It was going to link `trellis2`, but this
   project's default here is `BUILD_SHARED_LIBS=ON` and the vendored core has no
   `TRELLIS2_API` decoration, so its symbols are not exported. The tool compiles
   the vendored sources into its own target instead.
5. **CMake version ranges are unavailable.** `find_package(Eigen3 5.0...<6)`
   needs CMake 3.19; this project requires 3.14. And a single version would be
   wrong anyway — Eigen's config file treats it as an *exact* component match, so
   `Eigen3 5.0` rejects 5.0.1. Resolved with an unversioned `find_package` plus
   an explicit `Eigen3_VERSION` check.
6. **Running `./format_code.sh` also reformatted three pre-existing files** —
   `print_remesh.cpp`, `trellis2.cpp`, `fuzz/fuzz_image.cpp`, 10 lines total,
   all `( void ) x;` → `( void )x;` inside `#ifndef TRELLIS2_USE_CGAL` blocks.
   That is pre-existing formatter drift surfaced by this change, not caused by
   it. Left in the working tree for the reviewer to keep or drop; it is unrelated
   to this phase.
7. `*.cpp.tmp0` files in the working tree predate this session (timestamp 20:48)
   and were not touched.

## Verification

| Check | Result |
|---|---|
| Configure, no CGAL, vcpkg manifest | `AutoRemesher quad remeshing: enabled (Eigen 5.0.1)` |
| Build `build-quad` (clang, Ninja, Release) | success |
| Configure + build with `-DVCPKG_MANIFEST_FEATURES=cgal` | success, exit 0 — `CGAL Alpha Wrap print remeshing: enabled` **and** `AutoRemesher quad remeshing: enabled` |
| `./format_code.sh` | clean; `third_party/` untouched as intended |

The CGAL check is the important one: it confirms the switch to vcpkg manifest
mode did not drop CGAL, which was the one existing path this change could break.
CGAL now resolves from `build-cgalcheck/vcpkg_installed/x64-windows/share/cgal`
instead of the classic tree.

### Measurements

Clang/Release, `--target-quads 20000` for the dense meshes, `2000` otherwise.

| Input | Tris in | Wall | Peak RSS | Faces out | Quads | Boundary edges | Area retained |
|---|---|---|---|---|---|---|---|
| cube | 12 | 0.05 s | 7 MiB | 139 | 100.0 % | 0 (closed) | 94.5 % |
| sphere | 2 304 | 0.27 s | 26 MiB | 295 | 95.6 % | 0 (closed) | 99.2 % |
| dense sphere | 65 536 | 5.09 s | 236 MiB | 1 936 | 98.4 % | 0 (closed) | 99.9 % |
| dense sphere | 262 144 | 8.90 s | 281 MiB | 2 295 | 99.0 % | 0 (closed) | 99.9 % |

**Runtime is affordable.** 262 k triangles in 8.9 s at 281 MiB, and the cost
grows clearly sublinearly in triangle count (4× the triangles for 1.75× the
time). A 512³ dual-grid mesh should land in the tens of seconds — too slow for a
live preview, fine for an explicit export. Whether meshoptimizer pre-decimation
is *mandatory* is therefore still open, but it no longer looks like the make-or-
break question the plan assumed.

**`--target-quads` is a weak hint, not a target.** 20 000 requested produced
~2 300 faces. The mapping (`targetTriangleCount = quads * 2`, as upstream does)
feeds a voxel size, and adaptivity moves the result a long way from it. Phase 3
must not present this parameter as a face count.

### The important finding: non-manifold input

A mesh with one non-manifold vertex, a duplicated face and a dangling fin
retained **9.1 %** of its surface area, produced 78 boundary edges — and
`remesh()` still returned **success**. Isolating each defect:

| Input defect | Area retained |
|---|---|
| two disjoint components | 101.7 % — fine |
| duplicated face | 99.3 % — fine |
| dangling triangle fin | 99.5 % — fine |
| **one shared (non-manifold) vertex** | **49.9 %** — exactly one of two spheres survives |

The killer is the non-manifold *junction*, not duplicate faces or fins. This
inverts one of the Phase 3 items: the plan called for a position-key vertex weld,
which would *manufacture* precisely this defect. Phase 3 must split non-manifold
vertices and edges instead, and weld only where doing so cannot create a
junction. The plan has been corrected.

It also confirms risk 3 in the plan and makes it concrete: the failure is
**silent**, so the area-retention check built into the smoke tool has to become
part of the library path, not stay a diagnostic.

## Open points

- **No real 512³ pipeline mesh was measured.** `generations/` and `assets/` hold
  no mesh here. The dense spheres exercise triangle count but not the dual grid's
  non-manifold pathology, which — per the finding above — is the dimension that
  actually decides output quality. This needs a rerun on real data before Phase 3
  fixes a default `target_quads`.
- `docker/Dockerfile.demo` still has no Eigen 5.x; the Linux containers configure
  fine and report the stage as unavailable.
- The other Windows batch scripts do not yet pass the vcpkg toolchain file, so
  they build without the quad stage.
- Upstream import commit for `third_party/autoremesher/VERSION.md` is still
  `<to be pinned>` — imported from a local source drop, not a clone.
- Determinism is untested (Phase 5).

## Next phase

Phase 2 (tinybvh + own nearest-point traversal) is independent of this work and
is the cheapest remaining reduction of the GPL surface. Phase 3 depends on this
phase and now starts from a sharper problem statement for input preparation.

## Proposed commit message

```text
Vendor the AutoRemesher core and build it without Qt or TBB
```
