# AutoRemesher — MIT quad topology for the mid-poly export path

- **Status:** planned
- **GitHub Issue:** none yet — the roadmap `docs/PLAN.md` does not cover
  the export/print path, so an issue should be opened and this file
  renamed to `docs/plan/<issue>-autoremesher-quad-remesh.md`.
- **Branch:** `<issue>-autoremesher-quad-remesh` / optional
- **Area:** print_remesh / mesh_export / trellis2_capi / third_party /
  server / docs
- **Date:** 2026-08-11
- **Author:** AI (prepared for review)

## Goal

The CGAL path is currently the only way to get a *mid-poly mesh with a
usable texture* out of the pipeline: Alpha Wrap builds a watertight
enclosing surface, and `t2glb::mesh_to_projected_glb` rebakes the dense
per-vertex PBR onto its UV atlas. Two things are wrong with that:

1. **License.** CGAL's 3D Alpha Wrapping is GPL-3.0-or-later. The whole
   feature therefore lives behind `TRELLIS2_CGAL` and is absent from
   every default build (see `AGENTS.md`, *CGAL exception*).
2. **Topology.** Alpha Wrap emits a dense, isotropic triangle soup. It
   is watertight, but it is not the clean quad topology a user wants
   for retopo, subdivision, rigging, or a game asset.

[AutoRemesher](https://github.com/huxingyi/autoremesher) 1.0.0 is
MIT-licensed since its relicense release ("Relicense from GPLv3 to MIT
(reimplemented MIT-incompatible dependencies)", `CHANGELOGS.md`) and its
core library carries **no Qt, no CGAL, no geogram, no libigl** — only
Eigen, TBB, and the author's own MIT `isotropicremesher`. That makes it
the first realistic candidate for a *license-clean* mid-poly stage.

The goal of this plan is therefore:

- add a **quad remeshing stage** built on a vendored AutoRemesher core,
  encapsulated exactly like `print_remesh`, but with **no copyleft
  dependency** — its one external requirement, Eigen, is permissive
  enough to ship and trivial to install (D2);
- replace the CGAL AABB closest-surface query in `t2print::project_pbr`
  with vendored [tinybvh](https://github.com/jbikker/tinybvh) (MIT) plus
  an own nearest-point traversal, so the **texture rebake stops being
  the GPL-bound part** of the pipeline;
- keep the CGAL Alpha Wrap path unchanged as the watertight/printing
  backend, because AutoRemesher does **not** guarantee a closed surface.

## Non-goals

- Removing or deprecating the CGAL path. Alpha Wrap stays the
  watertight-for-print backend; this plan only ends its monopoly on
  "mid-poly + texture".
- Porting the AutoRemesher GUI, its Qt code, or its OpenGL preview. The
  parameter semantics we need are documented in AutoRemesher's own MIT
  `src/main.cpp` CLI, which is the only source we work from.

## Acceptance criteria

- [ ] A `cmake -B build -DCMAKE_BUILD_TYPE=Release` build with **no
      CGAL and no TBB**, with only `eigen3` present (vcpkg toolchain or
      `libeigen3-dev`), produces a working quad remesher.
- [ ] Without Eigen the build still succeeds; `TRELLIS2_AUTOREMESHER`
      reports itself unavailable instead of failing to configure.
- [ ] `mesh2glb in.t2mesh out.glb --quad <target_quads>` writes a GLB
      whose geometry comes from AutoRemesher and whose material comes
      from closest-surface projection of the dense source PBR.
- [ ] `t2print::project_pbr` works without CGAL; with CGAL present the
      tinybvh and CGAL backends agree within a documented tolerance on a
      fixture mesh.
- [ ] `t2_quad_remesh_available()` / `t2_prepare_quad_mesh()` exist,
      `T2_CAPI_ABI_VERSION` is bumped, and `server/engine.go` is
      propagated.
- [ ] No file under `third_party/` is reformatted; `./format_code.sh`
      produces an empty diff on the vendored tree.
- [ ] `THIRD_PARTY.md` lists the new components in its real sections
      instead of *Planned additions*, and `README.md` links to it.
- [ ] Existing `ctest --test-dir build -LE model` stays green; no tap in
      `docs/VERIFICATION.md` changes (this plan touches no neural stage).

## Context / affected files

| Path | Why relevant |
|---|---|
| `docs/ideas/autoremesher-master/src/AutoRemesher/` | the ~6.8k LOC MIT core to vendor (28 files) |
| `docs/ideas/autoremesher-master/thirdparty/isotropicremesher/` | 120 KB, MIT, same author; required by the core |
| `docs/ideas/autoremesher-master/thirdparty/eigen/` | not imported — but its `Eigen/Version` pins the known-good Eigen release (5.0.1) |
| `docs/ideas/autoremesher-master/src/main.cpp` | MIT CLI — authoritative source for parameter names, defaults, and ranges |
| `docs/ideas/tinybvh-main/tiny_bvh.h` | MIT, 397 KB single header, zero dependencies — the BVH builder for Phase 2 |
| `print_remesh.{h,cpp}` | `alpha_wrap` + `project_pbr`; the CGAL AABB tree in `project_pbr` is what Phase 2 replaces |
| `mesh_export.{h,cpp}` | `prepare_print_mesh`, `mesh_to_projected_glb`; gets a quad sibling |
| `trellis2_capi.h` / `.cpp` | new entry points → **ABI bump** |
| `examples/mesh2glb.cpp` | `--print` gets a `--quad` sibling |
| `server/engine.go`, `server/web/` | ABI propagation + optional viewer toggle |
| `CMakeLists.txt` | new `TRELLIS2_AUTOREMESHER` option, vendored sources |

## What AutoRemesher actually gives us (and what it does not)

Read from the sources, not from the README (the README still names
geogram/libigl; the 1.0.0 tree no longer contains them):

**Pipeline** (`AutoRemesher::remesh()`):
`initializeVoxelSize()` → `MeshSeparator::splitToIslands()` → per island
`resample()` (curvature-adaptive `IsotropicRemesher`) → `Parameterizer`
(frame field + mixed-integer least squares over Eigen sparse solvers) →
`QuadExtractor` → merge.

**API** (`src/AutoRemesher/autoremesher.h`): a plain
`AutoRemesher(vertices, triangles)` object with
`setTargetTriangleCount`, `setScaling`, `setGradientAdaptivity`,
`setAnisotropy`, `setSharpEdgeDegrees`, `setSmoothNormalDegrees`,
`setModelType(Organic|HardSurface)`, plus a C-style
`AutoRemesherProgressHandler(void* tag, float, const char*)`. Results:
`remeshedVertices()` / `remeshedQuads()`. This maps onto our existing
progress-callback style with no glue layer.

**Hard limitations to design around:**

- **Not watertight.** Islands whose parameterization throws are caught,
  logged, and *skipped* (`remesh()`, `catch (const std::exception&)`),
  and `QuadExtractor::extract()` returning false silently drops an
  island. Output can lose whole parts and can have boundaries. Nothing
  here replaces Alpha Wrap for printing.
- **Quad-*dominant*, not pure quad.** `remeshedQuads()` is
  `vector<vector<size_t>>`; the upstream CLI itself reports a
  "Non-quads" count. The export path must triangulate arbitrary
  polygons, not assume 4 indices.
- **Manifold expectations.** The isotropic remesher builds a halfedge
  mesh. Our dual-grid output is explicitly described as "heavily
  non-manifold" (`mesh_export.h`). Input preparation is not optional.
- **`std::cerr` logging.** The core prints timings and island failures
  straight to stderr. In the demo server that becomes log noise.
- **Cost is unknown to us.** Frame-field + MIQ parameterization on a
  512³ dual-grid mesh is not a millisecond operation. No runtime is
  claimed in this plan; Phase 1 measures it before anything is wired
  into the server.

## Architecture decisions

### D1 — Vendor the core, do not shell out to a binary

The obvious cheap route — write OBJ, run `autoremesher.exe`, read OBJ
back — is wrong here: the
upstream `main.cpp` constructs a `QApplication` and a `MainWindow` even
in `--input/--output` headless mode, so the "CLI" needs Qt and a usable
display/offscreen platform. Vendoring the core is both lighter and
license-clean: `src/AutoRemesher/` (28 files, 6809 LOC) plus
`thirdparty/isotropicremesher/` (120 KB) have zero Qt includes.

Target layout, consistent with `third_party/xatlas|meshoptimizer|stb`:

```
third_party/autoremesher/
├── LICENSE                  # MIT, Dust3D Project / Jeremy HU
├── VERSION                  # upstream commit/tag + date of import
├── PATCHES.md               # every local modification, with rationale
├── include/AutoRemesher/…   # the extensionless umbrella headers
├── src/…                    # the 28 core sources
└── isotropicremesher/…      # bundled MIT dependency, LICENSE included
```

Vendored code is **never reformatted** (`format_code.sh` already
excludes `third_party/`) and local edits stay minimal and listed in
`PATCHES.md`.

### D2 — Eigen via vcpkg, never vendored

`Parameterizer`, `FrameField`, `ConstrainedLeastSquares`, and
`MixedIntegerLeastSquares` use `Eigen/Dense`, `Eigen/Sparse`,
`Eigen/SparseCholesky`, `Eigen/SparseLU`, `Eigen/SparseQR`,
`Eigen/Eigenvalues`. Eigen cannot be designed away without rewriting the
solver core, which is out of scope.

Eigen is **MPL2** (`COPYING.README`), which is not on the
"MIT/BSD/Azure/PD-compatible" list in `AGENTS.md`. **Decision: Eigen is
not vendored — it is consumed as an external dependency via vcpkg.**

This is not a new pattern for the project, it is the existing one.
`cmake-ninja-win64-rocm-cgal.bat` already passes
`-DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake`, and
the current `build/CMakeCache.txt` resolves CGAL, Boost, GMP and MPFR
out of `C:/vcpkg/installed/x64-windows`. Eigen slots into exactly that
mechanism:

```sh
vcpkg install eigen3      # header-only, no transitive deps; currently 5.0.1
```

```cmake
# NOT find_package(Eigen3 3.4 ...): Eigen's own Eigen3ConfigVersion.cmake
# treats a single version as an exact component match ("3.4 matches 3.4.0.0
# to <3.5.0.0"), so a 3.4 request fails against 5.0.1. Use a range.
find_package(Eigen3 3.4...<6 CONFIG QUIET)     # target: Eigen3::Eigen
```

**Known-good version: 5.0.1.** The upstream AutoRemesher tree bundles
Eigen 5.0.1 itself (`thirdparty/eigen/Eigen/Version`:
`EIGEN_MAJOR_VERSION 5`, `EIGEN_MINOR_VERSION 0`,
`EIGEN_PATCH_VERSION 1`), and `vcpkg install eigen3` currently resolves
to exactly `eigen3:x64-windows@5.0.1`. So the Windows path builds
against the same version upstream develops against — no API drift.

**Linux is the unverified half.** Debian/Ubuntu `libeigen3-dev` ships
**3.4.0**, not 5.x, and whether the AutoRemesher core compiles against
3.4 has not been tested by us or (visibly) by upstream. The range above
is therefore optimistic on purpose: Phase 1 compiles against both 5.0.1
and 3.4.0 and settles it. If 3.4 fails, the floor is raised to
`5.0...<6` and Linux needs vcpkg or a manual Eigen — which must then be
written into `docker/Dockerfile.demo` rather than papered over with the
distro package.

**What this buys and what it does not.** It keeps every MPL2 file out
of the MIT repository, so the `third_party/` rule in `AGENTS.md` holds
literally and no license text has to be relaxed. It does *not* remove
MPL2 code from the shipped binary — Eigen templates still compile into
`libtrellis2`. The residual MPL2 obligation on binary distribution is
to retain the notice and keep Eigen's source obtainable; since we never
modify Eigen, a URL in `THIRD_PARTY.md` discharges it, and MPL2 §3.3
explicitly permits the combination with a differently-licensed Larger
Work. For proportion: the `TRELLIS2_CGAL` path already links GPL-3 CGAL
plus LGPL GMP/MPFR through the same vcpkg install, so MPL2 is strictly
the lighter burden of the two.

**Cost, stated honestly.** `TRELLIS2_AUTOREMESHER` becomes a probe like
`TRELLIS2_CGAL` and switches itself off when Eigen is absent, so the
feature is not unconditionally present in an arbitrary clone. That was
the objection against the non-vendored route in the first draft, and it
does not fully disappear — it only shrinks: `eigen3` is header-only
with no transitive dependencies and is packaged everywhere, whereas
CGAL drags Boost, GMP and MPFR behind it. Missing Eigen is a one-line
install, missing CGAL is an afternoon.

Independent of the sourcing decision: the LGPL-adjacent optional Eigen
backends (`EIGEN_USE_MKL_ALL`, SuperLU, UmfPack, Pardiso) stay off, and
the Apple-only `Eigen/AccelerateSupport` include in the upstream tree is
dropped together with the macOS `NL_USE_BLAS` block.

Open sub-decision for Phase 0: the project currently uses vcpkg in
**classic mode** (`VCPKG_MANIFEST_MODE:BOOL=OFF`, packages installed
globally under `C:/vcpkg`). A `vcpkg.json` manifest with a
`builtin-baseline` would pin CGAL and Eigen versions reproducibly, but
it changes the author's established workflow, so it is offered rather
than assumed.

### D3 — Drop TBB, ship a small shim

TBB would be a second new build dependency (the bundled copy is TBB
2017 and wants its own CMake build). The actual usage surface is tiny —
`tbb::parallel_for` over `tbb::blocked_range<size_t>`, one
`tbb::combinable` in `parameterizer.cpp`, and one `tbb/mutex.h`
include. A ~150-line MIT header
`third_party/autoremesher/t2_tbb_shim.h` implementing exactly those
four names over `std::thread` (which the project already links via
`Threads::Threads` for xatlas) removes the dependency entirely. The
shim is our own code, listed in `PATCHES.md`, and the four upstream
`#include <tbb/…>` / `<oneapi/tbb/…>` lines are redirected to it.

Escape hatch: `TRELLIS2_AUTOREMESHER_TBB=ON` uses a real
`find_package(TBB)` instead, for benchmarking the shim against upstream
scheduling.

### D4 — Alpha Wrap and AutoRemesher are complementary, not rivals

Given D-limitations above, the export path becomes a small matrix that
the caller chooses from:

| Mode | Manifold-ization | Topology | Watertight? | License |
|---|---|---|---|---|
| `dense` (today's default) | component filter | preserved | no | MIT |
| `print` (today's `--print`) | CGAL Alpha Wrap | dense triangles | yes | GPL path |
| `quad` (new) | component filter + own cleanup | AutoRemesher quads | **no** | MIT |
| `print+quad` (new, opt-in) | CGAL Alpha Wrap | AutoRemesher quads | not guaranteed after remesh | GPL path |

`print+quad` is the highest-quality combination when CGAL is present
(Alpha Wrap hands AutoRemesher a clean 2-manifold, which is exactly the
input it wants), but it must not be advertised as watertight — the
quad stage can reintroduce boundaries. The UI/CLI wording has to say so.

### D5 — tinybvh as the builder, own traversal for the query

`t2print::project_pbr` is a closest-point-on-triangle-soup query plus
barycentric interpolation of six channels. Everything CGAL-specific in
it is `AABB_tree::closest_point_and_primitive`.

**tinybvh** (MIT, Jacco Bikker, v1.7.1) is the right foundation:
`tiny_bvh.h` is a 397 KB single header with **zero dependencies**,
C++11, `TINYBVH_IMPLEMENTATION` in exactly one TU — the same vendoring
shape as `stb` and `xatlas`. It brings a binned-SAH builder, an AVX
builder, a spatial-splits `BuildHQ`, and a built-in threaded build.

**But it does not have a closest-point query.** Its query surface is
`Intersect` / `IsOccluded` / `Intersect256Rays`, i.e. rays, plus
`IntersectSphere` — which is a boolean *overlap* test, not a nearest
lookup, and must not be mistaken for one. So tinybvh replaces the
*builder*, not the query.

That is fine, because the data we need is public. `BVH::bvhNode` (Wald
32-byte nodes, root at index 0), `BVH::primIdx`, `BVH::vertIdx`, and
`BVH::verts` are all public members — the `private:` before them in
`tiny_bvh.h` is commented out. A nearest-point traversal (best-first
descent over `bvhNode`, point-AABB lower bound for pruning,
point-triangle distance at the leaves, running upper bound) sits on top
of a built `BVH` in roughly 80–100 lines **without patching the
vendored header**. That is our own code, written from the standard
geometric formulation — not transcribed from CGAL, and not from
AutoRemesher's `AxisAlignedBoudingBoxTree` either, which does box-pair
intersection rather than nearest-point queries.

Feeding it: use the **indexed** build
`Build(const bvhvec4slice&, const uint32_t* indices, primCount)`, which
matches our `verts`/`tris` layout directly. Expand positions from
`float[3]` to `bvhvec4` (16 B/vertex) rather than using a
12-byte-strided `bvhvec4slice` — `operator[]` reads a full `bvhvec4`,
so a 12-byte stride overreads 4 bytes past the last vertex. Expanding
per vertex also costs far less than the non-indexed path, which would
expand per triangle corner.

Selection stays a build-time detail behind the unchanged
`t2print::project_pbr` signature: CGAL backend when
`TRELLIS2_USE_CGAL`, tinybvh otherwise, with
`TRELLIS2_FORCE_TINYBVH=ON` to force it inside a CGAL build so the two
can be compared in one binary.

**Beyond parity — the reason this is worth more than a CGAL swap:**
tinybvh gives the project *ray* queries, which the CGAL path never
offered. Upstream TRELLIS does its texture reprojection with cuBVH ray
casting, and a normal-offset raycast bake is materially better than a
nearest-point bake on thin geometry, where "closest surface" happily
picks the material from the wrong side of a wall. Scope discipline:
Phase 2 ships nearest-point parity only; the raycast bake is written up
as a follow-up so it can be judged on its own bake results.

### D6 — Encapsulation mirrors `print_remesh`

New module `quad_remesh.{h,cpp}`:

```cpp
namespace t2quad {

TRELLIS2_API bool available();

struct QuadRemeshOptions {
    int    target_quads   = 20000; // upstream CLI default is 50000
    float  edge_scaling   = 1.0f;  // 1.0 – 4.0
    float  adaptivity     = 1.0f;  // 0.0 – 1.0, curvature-adaptive density
    float  anisotropy     = 1.0f;  // 0.0 – 1.0, curvature-adaptive elongation
    float  sharp_edge_deg = 90.0f; // 30 – 180
    float  smooth_normal_deg = 0.0f;
    bool   hard_surface   = false; // ModelType::HardSurface
};

// Quad-dominant output: `faces` is a flat index stream, `face_sizes` gives
// the vertex count per face (4 for quads, 3 and n for the rest).
TRELLIS2_API bool remesh(const std::vector<float>   & verts,
                         const std::vector<int32_t> & tris,
                         const QuadRemeshOptions    & opt,
                         std::vector<float>         & out_verts,
                         std::vector<int32_t>       & out_faces,
                         std::vector<int32_t>       & out_face_sizes,
                         std::string                & err);

} // namespace t2quad
```

Same contract as `print_remesh.h`: no vendored type in the public
header, builds without the backend and reports `available() == false`.
`available()` is `true` in any build where the vendored sources are
compiled in, i.e. by default — that is the whole point.

Everything downstream (GLB, `.t2mesh`, the C API) keeps taking
triangles, so `mesh_export` triangulates on the way out (fan for
triangles, shorter-diagonal split for quads, fan for n-gons) while the
quad stream stays available for an OBJ/quad-preserving export later.

### D7 — Format and API rules that this plan must respect

- `T2_CAPI_ABI_VERSION` bump + `server/engine.go` propagation for every
  C-API change (`AGENTS.md`, and the header's own rule).
- New exported functions carry `TRELLIS2_API` / `TRELLIS2_CAPI`;
  helpers stay `static`/anonymous.
- CMake option prefix `TRELLIS2_`.
- Sources formatted with `./format_code.sh` (clang-format 18.1.8);
  `third_party/` excluded.
- No `.t2mesh` version bump in this plan — quads are triangulated
  before they reach the container. A quad-preserving `T2MESH04` would
  be its own plan.

## Plan (phases)

### Phase 0 — Decisions (blocking, human)

- [ ] Confirm **Eigen from vcpkg / `libeigen3-dev`**, not vendored
      (D2), and accept that `TRELLIS2_AUTOREMESHER` auto-disables when
      Eigen is absent.
- [ ] Decide vcpkg **classic vs. manifest mode** (`vcpkg.json` with a
      `builtin-baseline` pinning `cgal` and `eigen3`).
- [ ] Confirm the upstream import points (commit/tag) to pin in
      `third_party/autoremesher/VERSION` and
      `third_party/tinybvh/VERSION`.
- [ ] Review the new root `THIRD_PARTY.md`. It documents the current
      state; the components of this plan are listed there as *planned*
      until their phase lands. (`xatlas`, `meshoptimizer`, and `stb`
      have no separate `LICENSE` file, but each retains its upstream
      notice inside the vendored file, so MIT/PD attribution is
      satisfied — a per-directory `LICENSE` would be hygiene, not a
      fix.)

Nothing is imported before this is answered.

### Phase 1 — Vendor and build in isolation

- [ ] Import `src/AutoRemesher/` + `include/AutoRemesher/` +
      `thirdparty/isotropicremesher/` into
      `third_party/autoremesher/`, with `LICENSE`, `VERSION`,
      `PATCHES.md`.
- [ ] Add `t2_tbb_shim.h` (D3) and redirect the four TBB includes.
- [ ] Drop the Apple `AccelerateSupport` include and the `NL_USE_BLAS`
      assumption.
- [ ] Route `std::cerr` through a single `AR_LOG(...)` macro that is
      silent unless `TRELLIS2_AUTOREMESHER_VERBOSE` is set (D-limitation
      "std::cerr logging"). Document each touched line in `PATCHES.md`.
- [ ] `find_package(Eigen3 3.4...<6 CONFIG QUIET)` → link
      `Eigen3::Eigen` `PRIVATE` (D2). No Eigen file is copied into the
      repository.
- [ ] Compile the core against **both** Eigen 5.0.1 (vcpkg, upstream's
      own bundled version) and 3.4.0 (Debian/Ubuntu `libeigen3-dev`).
      Record the outcome; raise the CMake floor to `5.0...<6` if 3.4
      does not build.
- [ ] `TRELLIS2_AUTOREMESHER` CMake option (default `ON`, but
      auto-disabled with a `message(STATUS ...)` when Eigen is missing,
      mirroring the existing `TRELLIS2_CGAL` probe), sources added to
      the `trellis2` target, `/bigobj` on MSVC (upstream needs it),
      warnings suppressed for the vendored tree only.
- [ ] Add `eigen3` to the documented vcpkg install line, extend the
      Windows batch scripts that should build the feature with the
      vcpkg toolchain file, and give `docker/Dockerfile.demo` whichever
      Eigen the compile matrix above proves usable.
- [ ] Standalone smoke tool `examples/quad_remesh_cli.cpp`: OBJ in,
      OBJ out — proves the core builds and runs before any wiring.
- [ ] **Measure** on a cube, a sphere, and a real 512³ pipeline mesh:
      wall time, peak RSS, quad/non-quad ratio, boundary-edge count,
      dropped islands. These numbers, not guesses, decide the default
      `target_quads` and whether a meshoptimizer pre-decimation is
      mandatory.

### Phase 2 — tinybvh, so the texture path stops needing CGAL

- [ ] Vendor `third_party/tinybvh/tiny_bvh.h` + `LICENSE` + `VERSION`
      (v1.7.1 plus import commit). Header only — none of the demos,
      `tiny_ocl.h`, `testdata/`, or the OpenCL kernels are imported.
- [ ] `TINYBVH_IMPLEMENTATION` in exactly one TU. Candidate: a small
      `third_party/tinybvh/tiny_bvh_impl.cpp` added to the `trellis2`
      target, so `print_remesh.cpp` stays a plain consumer and a second
      user (raycast bake, AO) does not cause duplicate symbols.
- [ ] Nearest-point traversal over `BVH::bvhNode` / `primIdx` (D5) as
      an own helper in `print_remesh.cpp` — vendored header unpatched.
- [ ] `t2print::project_pbr` gets the tinybvh backend, selected by
      `#ifdef`; `TRELLIS2_FORCE_TINYBVH` for A/B testing in a CGAL
      build.
- [ ] Check tinybvh's build threading against the query threading
      already in `project_pbr` so the two do not oversubscribe; disable
      `threadedBuild` if it does.
- [ ] `tests/test_print_remesh.cpp`: extend so the tinybvh backend runs
      unconditionally (no more 77-skip for the projection half) and,
      when CGAL is present, both backends are compared against each
      other with an explicit tolerance. Include a degenerate/duplicate
      triangle fixture — CGAL rejects degenerate primitives outright
      (`tri.is_degenerate()` skip), tinybvh does not, so the two must
      be shown to agree on the same filtered input.
- [ ] Benchmark both backends: build time and queries/s on a real
      pipeline mesh.
- [ ] `mesh_to_projected_glb` loses its "always CGAL" restriction; its
      header comment in `mesh_export.h` and the `t2_bake_projected_glb`
      doc block in `trellis2_capi.h` ("Returns NULL when CGAL support is
      unavailable") are corrected.

This phase is independently valuable and does not depend on Phase 1 —
it can be reviewed and merged on its own, and it is the cheapest real
reduction of the GPL surface in this plan.

### Phase 3 — The `quad_remesh` module

- [ ] `quad_remesh.{h,cpp}` per D6, with input validation matching
      `print_remesh.cpp`'s style (finite coordinates, index range,
      degenerate triangle skip, non-empty bbox).
- [ ] Input preparation for the non-manifold dual-grid mesh: vertex
      weld by position key, degenerate-triangle removal, duplicate-face
      removal, and — behind an option — a meshoptimizer decimation to a
      configurable input triangle budget.
- [ ] Progress: bridge `AutoRemesherProgressHandler` to the existing
      stage-progress callback style.
- [ ] Failure policy: report *how many* islands were dropped in `err`
      instead of letting them vanish into stderr; a run that loses more
      than a configurable fraction of the input area is an error, not a
      success.
- [ ] `t2glb::prepare_quad_mesh(...)` in `mesh_export.{h,cpp}`:
      component filter → quad remesh → triangulation → the existing
      projected bake.
- [ ] `mesh2glb --quad [target_quads]`, combinable with `--print`
      (D4 matrix), including the honest "not guaranteed watertight"
      message.

### Phase 4 — C-API, server, viewer

- [ ] `t2_quad_remesh_available()` and `t2_prepare_quad_mesh(...)` in
      `trellis2_capi.{h,cpp}`, documented in the header's existing
      style.
- [ ] **Bump `T2_CAPI_ABI_VERSION`** and propagate in
      `server/engine.go`.
- [ ] Server: expose the mode as a job parameter; keep the single
      inference mutex semantics; make sure a long quad remesh cannot
      block progress polling.
- [ ] Viewer: a mode selector next to the existing export options; no
      CDN, no build step.

### Phase 5 — Validation and documentation

- [ ] `tests/test_quad_remesh.cpp` (label: no model assets required):
      cube/sphere fixtures, deterministic-output check (two runs on the
      same input must be byte-identical — the merge is island-ordered,
      so this should hold and is worth asserting), quad-ratio floor,
      index-range and finiteness invariants, and an explicit test that
      documents that boundary edges *may* exist.
- [ ] Benchmark table in `docs/architecture/` with the Phase 1
      numbers: mode × input size × time × triangle/quad count.
- [ ] `AGENTS.md`: third-party list, MPL2 decision, and the new CMake
      options.
- [ ] Move AutoRemesher, isotropicremesher, tinybvh, and Eigen from
      the *Planned additions* table of `THIRD_PARTY.md` into the real
      sections, and link `THIRD_PARTY.md` from the `## License` section
      of `README.md`.
- [ ] `docs/progress/<key>_<phase>-autoremesher-quad-remesh.md` per
      completed phase, `Commits: <to be added after review>`.

## Tests / verification

- `cmake -B build-quad -DCMAKE_BUILD_TYPE=Release -DTRELLIS2_CGAL=OFF`
  then `cmake --build build-quad -j` — proves the feature is genuinely
  CGAL-free.
- `ctest --test-dir build-quad -LE model` — new quad and BVH tests run
  without assets; no `SKIP`/77 counted as a pass.
- With CGAL: `ctest --test-dir build` — the CGAL-vs-tinybvh comparison
  test must pass.
- `./format_code.sh && git diff --stat` — empty diff under
  `third_party/`.
- Manual: `mesh2glb sample.t2mesh out_quad.glb --quad 20000` and
  `--print --quad`, inspected in the viewer and in Blender for quad
  flow, seams, and material fidelity against the `--print` result.
- `docs/VERIFICATION.md` is untouched: no neural stage is affected.

## Risks / open questions

1. **Runtime.** Frame-field parameterization on a full-resolution
   dual-grid mesh may be far too slow for the interactive server path.
   Mitigation: mandatory meshoptimizer pre-decimation, a target-quad
   cap, and — if the Phase 1 numbers are bad — offering the mode only
   as an explicit download-time operation rather than in the live
   preview. Decide with measurements, not now.
2. **Robustness on non-manifold input.** This is the most likely source
   of hangs and crashes, since our geometry is exactly the kind the
   halfedge remesher dislikes. Mitigation: strict input preparation
   (Phase 3), plus fuzz coverage — the existing `fuzz/` harnesses cover
   untrusted image input, and a mesh-level harness for `t2quad::remesh`
   would be a reasonable follow-up (its own plan).
3. **Silent quality loss.** Dropped islands currently disappear into
   stderr. The Phase 3 failure policy exists specifically to stop a
   half-remeshed model from looking like a success.
4. **Eigen availability and version, not Eigen license.** With D2 the
   license question is settled (external, documented in
   `THIRD_PARTY.md`). Two practical risks remain: a clone without
   vcpkg/`libeigen3-dev` silently builds without the quad stage — the
   `message(STATUS ...)` line and `t2quad::available()` have to make
   that obvious rather than puzzling — and Linux distros ship Eigen
   **3.4.0** while upstream AutoRemesher develops against **5.0.1**.
   The Phase 1 compile matrix decides whether the Linux path works at
   all or needs vcpkg there too.
5. **Repository growth** is now small — `tiny_bvh.h` (397 KB) plus the
   AutoRemesher core (~450 KB). No multi-megabyte header tree.
6. **Determinism** is expected from reading the merge code but has not
   been verified; Phase 5 asserts it. If it does not hold, the parallel
   island scheduling has to be made order-stable.
7. **`print+quad` and watertightness.** Users will assume the
   combination is printable. It is not guaranteed. Wording in CLI,
   server, and viewer must be checked at review time.
8. **Upstream drift.** AutoRemesher 1.0.0 is recent; the vendored copy
   must record its import point so a later update is a diffable
   operation and not an archaeology exercise. Same for tinybvh, whose
   public-member layout (`bvhNode`, `primIdx`) our traversal depends on
   — that is a deliberate, documented coupling to an unstable surface,
   and a version bump has to be reviewed, not applied blindly.
9. **Numerical differences to the CGAL bake.** CGAL builds its tree
   over exact-predicate `Triangle_3` primitives and skips degenerates;
   tinybvh is plain float. Expect small disagreements at triangle
   edges and on slivers. The tolerance in the comparison test has to be
   justified by measurement, and if a fixture shows a *visible* seam
   difference in the bake, that is a finding, not a tolerance to widen.
10. **tinybvh binary size and SIMD.** The header pulls in AVX/NEON
    paths; check the effect on the shared-library size the Go server
    dlopens, and make sure no build-time `-mavx2` requirement leaks
    into the project's baseline flags.

## Follow-ups (own plans, not in scope here)

- **Raycast texture bake.** With tinybvh present, replace the
  nearest-point transfer with a normal-offset raycast bake (the
  approach upstream takes with cuBVH). Better on thin geometry; needs
  its own quality comparison before it becomes the default.
- **Quad-preserving container.** A `T2MESH04` that carries quads
  instead of triangulating them at export.
- **Mesh-level fuzz harness** for `t2quad::remesh`.

## Release note

Adds an MIT quad remeshing stage (vendored AutoRemesher core) and a
tinybvh-based closest-surface backend, so mid-poly meshes with
projected PBR materials no longer require a GPL CGAL build.

## Proposed commit message

Phase 1 (the first reviewable unit) would be committed as:

```text
Vendor the AutoRemesher core and build it without Qt or TBB
```
