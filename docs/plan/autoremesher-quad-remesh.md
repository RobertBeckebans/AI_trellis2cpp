# AutoRemesher — MIT quad topology for the mid-poly export path

- **Status:** planned
- **GitHub Issue:** none — deliberately tracked by plan key
  (`autoremesher-quad-remesh`) instead, so progress entries use
  `docs/progress/autoremesher-quad-remesh_<phase>-<short-name>.md`.
- **Branch:** `autoremesher-quad-remesh` / optional
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

**Decided: newest Eigen, Windows first.** The upstream AutoRemesher
tree bundles Eigen **5.0.1** itself (`thirdparty/eigen/Eigen/Version`:
`EIGEN_MAJOR_VERSION 5`, `EIGEN_MINOR_VERSION 0`,
`EIGEN_PATCH_VERSION 1`), and `vcpkg install eigen3` resolves to exactly
`eigen3:x64-windows@5.0.1`. Building against the same version upstream
develops against removes the API-drift question entirely, so that is
what we do rather than stretching for backwards compatibility:

```cmake
# A range, not a single version: Eigen's own Eigen3ConfigVersion.cmake
# treats a single requested version as an exact component match
# ("3.4 matches 3.4.0.0 to <3.5.0.0"), so find_package(Eigen3 5.0) would
# reject a later 5.1. The range accepts the whole 5.x line.
find_package(Eigen3 5.0...<6 CONFIG QUIET)     # target: Eigen3::Eigen
```

**Consequence for Linux, stated rather than hidden.** Debian/Ubuntu
`libeigen3-dev` ships 3.4.0, which this floor excludes. The Linux
containers in `docker/` therefore do not get the quad stage until they
either install Eigen 5.x explicitly or use vcpkg as well. That is a
deliberate deferral, not an oversight: the probe auto-disables the
feature, so a Linux build stays green and simply reports the stage as
unavailable. Nothing else in the pipeline regresses. Raising Linux to
Eigen 5 is a follow-up whenever the Linux demo needs the stage.

**Decided: vcpkg manifest mode.** A `vcpkg.json` in the project root
pins the versions reproducibly instead of relying on whatever happens to
sit in the global `C:/vcpkg/installed` tree:

```json
{
  "name": "trellis2-cpp",
  "builtin-baseline": "<vcpkg registry commit>",
  "dependencies": [ { "name": "eigen3", "version>=": "5.0.1" } ],
  "features": {
    "cgal": {
      "description": "CGAL Alpha Wrap print remeshing (GPL-3.0-or-later)",
      "dependencies": [ "cgal" ]
    }
  }
}
```

Putting CGAL behind a **manifest feature** rather than a plain
dependency mirrors the encapsulation the code already enforces: a
default configure resolves only Eigen, and only
`-DVCPKG_MANIFEST_FEATURES=cgal` (in
`cmake-ninja-win64-rocm-cgal.bat`) pulls in CGAL, Boost, GMP and MPFR.
The copyleft dependency then cannot be dragged in by accident.

**Migration warning.** Manifest mode is not a no-op for existing
builds. As soon as `vcpkg.json` sits next to `CMakeLists.txt`, any
configure using the vcpkg toolchain switches to manifest mode
(`VCPKG_MANIFEST_MODE` is currently `OFF`), installs into
`<build>/vcpkg_installed`, and **ignores the classic
`C:/vcpkg/installed` tree**. The CGAL build therefore loses CGAL unless
`-DVCPKG_MANIFEST_FEATURES=cgal` is added to that script in the same
change. This is a one-line edit per script, but it must land together
with `vcpkg.json`, not after it.

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
`TRELLIS2_CGAL` and switches itself off when Eigen 5.x is absent, so the
feature is not unconditionally present in an arbitrary clone. That was
the objection against the non-vendored route in the first draft and it
does not disappear — with the 5.0 floor it is in fact sharper than the
"any Eigen" version would have been, since distro packages no longer
qualify. What keeps it acceptable is that the manifest resolves it
automatically on the author's Windows path: `eigen3` is header-only
with no transitive dependencies, so vcpkg fetches it in seconds,
whereas CGAL drags Boost, GMP and MPFR behind it.

Independent of the sourcing decision: the LGPL-adjacent optional Eigen
backends (`EIGEN_USE_MKL_ALL`, SuperLU, UmfPack, Pardiso) stay off, and
the Apple-only `Eigen/AccelerateSupport` include in the upstream tree is
dropped together with the macOS `NL_USE_BLAS` block.

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

### Phase 0 — Decisions (settled)

Recorded here so the phases below do not re-open them:

- **Eigen is external, never vendored** (D2), taken as the newest
  release vcpkg offers — currently 5.0.1, the same version upstream
  AutoRemesher bundles. CMake floor `5.0...<6`.
- **vcpkg manifest mode** with a `vcpkg.json` in the project root, CGAL
  behind a manifest *feature* so a default configure never resolves it.
  The `-DVCPKG_MANIFEST_FEATURES=cgal` edit to
  `cmake-ninja-win64-rocm-cgal.bat` ships in the same change.
- **Windows first.** The Linux containers keep building; they simply
  report the quad stage as unavailable until they get Eigen 5.x.
- **No GitHub issue.** Tracked by plan key only; the file is not
  renamed.

Remaining before Phase 1 imports anything: pin the upstream import
points (commit/tag) for `third_party/autoremesher/VERSION` and
`third_party/tinybvh/VERSION`.

`THIRD_PARTY.md` in the repository root documents the current state and
carries this plan's components under *Planned additions* until their
phase lands. (`xatlas`, `meshoptimizer`, and `stb` have no separate
`LICENSE` file, but each retains its upstream notice inside the vendored
file, so MIT/PD attribution is satisfied — a per-directory `LICENSE`
would be hygiene, not a fix.)

### Phase 1 — Vendor and build in isolation

Results: `docs/progress/autoremesher-quad-remesh_phase1-vendor-core.md`.

- [x] Import `src/AutoRemesher/` + `include/AutoRemesher/` +
      `thirdparty/isotropicremesher/` into
      `third_party/autoremesher/`, with `LICENSE`, `VERSION.md`,
      `PATCHES.md`. (`VERSION` without a suffix collides with the
      standard `<version>` header on a case-insensitive filesystem.)
- [x] `t2_tbb_shim.h` (D3) — reached through `compat/tbb/*.h`, so the
      upstream sources are **not** patched for TBB at all.
- [x] Apple `AccelerateSupport`: no patch needed, it is already behind
      `#if AUTO_REMESHER_USE_ACCELERATE`, which we never define.
- [x] Route `std::cerr` through `T2_AR_LOG`, silent unless
      `TRELLIS2_AUTOREMESHER_VERBOSE` is set. Applied as a replayable
      `sed`, documented in `PATCHES.md`.
- [x] `vcpkg.json` (D2) plus `-DVCPKG_MANIFEST_FEATURES=cgal` in
      `cmake-ninja-win64-rocm-cgal.bat`, in the same change.
- [x] Eigen through `find_package(Eigen3 CONFIG QUIET)` + an explicit
      5.x version check, linked `PRIVATE`. No Eigen file in the tree.
      (A version *range* would need CMake 3.19; this project targets
      3.14. Requesting `Eigen3 5.0` would fail against 5.0.1 because
      Eigen's config treats a single version as an exact match.)
- [x] `TRELLIS2_AUTOREMESHER` option with the `TRELLIS2_CGAL`-style
      probe; vendored sources built with `-w` / `/w /bigobj` and with
      `_USE_MATH_DEFINES;NOMINMAX`, which upstream sets globally in
      `autoremesher.pro` and without which `M_PI` is undefined on MSVC.
- [x] Standalone smoke tool `examples/quad_remesh_cli.cpp`. It compiles
      the vendored sources into its own target rather than linking
      `trellis2`, because the core has no `TRELLIS2_API` decoration and
      the project's default build here is `BUILD_SHARED_LIBS=ON`.
- [x] **Measured** on cube, sphere, and dense spheres up to 262 k
      triangles, plus an isolated non-manifold defect matrix. Numbers in
      the progress doc; the headline is that runtime is affordable
      (262 k tris → 8.9 s, 281 MiB) and that a single non-manifold
      vertex silently costs half the model while `remesh()` still
      returns success.
- [ ] Verify the CGAL build still configures and links after the
      manifest switch — the one existing path this change can break.
- [ ] Re-measure on a **real 512³ pipeline mesh**. Not possible here:
      `generations/` and `assets/` hold no mesh, so the dense spheres
      are a synthetic proxy that exercises triangle count but not the
      dual grid's non-manifold pathology.
- [ ] Add `eigen3` to the documented vcpkg install line, extend the
      other Windows batch scripts with the vcpkg toolchain file, and
      decide what `docker/Dockerfile.demo` does about Eigen 5.x.

### Phase 2 — tinybvh, so the texture path stops needing CGAL

Results: `docs/progress/autoremesher-quad-remesh_phase2-tinybvh-projection.md`.

- [x] Vendor `third_party/tinybvh/tiny_bvh.h` + `LICENSE` + `VERSION.md`
      (v1.7.1). Header only — no demos, no `tiny_ocl.h`, no `testdata/`,
      no OpenCL kernels. The header itself is unmodified.
- [x] `TINYBVH_IMPLEMENTATION` in exactly one TU
      (`third_party/tinybvh/tiny_bvh_impl.cpp`), in its own
      `trellis2_tinybvh` target: tiny_bvh.h's implementation half needs
      C++17 (`std::scoped_lock`) while the library stays C++14, which
      `AGENTS.md` reserves C++17 for the CGAL path.
- [x] Nearest-point traversal over `BVH::bvhNode` / `primIdx`, vendored
      header unpatched. A stack-depth guard reports failure rather than
      dropping a subtree and returning a plausible wrong answer.
- [x] `t2print::project_pbr` gets the tinybvh backend;
      `TRELLIS2_FORCE_TINYBVH` selects it inside a CGAL build. Both
      backends are compiled whenever CGAL is present and reachable via
      `project_pbr_backend()`, so the comparison needs one binary, not two.
- [x] Build threading checked: the surface is fully built before any query
      worker starts, so tinybvh's build pool never overlaps the query pool.
- [x] `tests/test_print_remesh.cpp` builds and runs unconditionally now,
      covering barycentric correctness, a degenerate/duplicate fixture, and
      the cross-backend comparison. `tests/test_mesh_export.cpp` asserted
      that a CGAL-free projected bake must *fail* — inverted, since that is
      exactly what this phase fixes.
- [x] `mesh_to_projected_glb` no longer gates on CGAL; the stale claims in
      `mesh_export.h` and the `t2_bake_projected_glb` doc block in
      `trellis2_capi.h` are corrected. Comment-only C-API change, so no
      `T2_CAPI_ABI_VERSION` bump.
- [x] **Cross-backend comparison run** (2026-08-12, ROCm/HIP build):
      `max |tinybvh - cgal| = 9.77e-05` over 24576 samples, inside the
      proposed 1e-4 tolerance. The earlier "does not compile" blocker was
      specific to the system LLVM clang; ROCm's clang builds the CGAL path
      without complaint, which also corrects the Phase 1 note.
- [ ] Benchmark both backends against each other: build time and queries/s.
      Only their agreement is measured, not their relative cost.

This phase is independently valuable and does not depend on Phase 1 —
it can be reviewed and merged on its own, and it is the cheapest real
reduction of the GPL surface in this plan.

### Phase 3 — The `quad_remesh` module

Results: `docs/progress/autoremesher-quad-remesh_phase3-quad-module.md`.

- [x] `quad_remesh.{h,cpp}` per D6, with input validation in
      `print_remesh.cpp`'s style.
- [x] Input preparation: drop zero-area and duplicate faces, optional
      `meshopt_simplify` to an input triangle budget, split non-manifold
      vertices. **Correction to the finding this phase was rewritten for:**
      with degenerate removal in place the splitter changes nothing on the
      synthetic fixtures — the zero-area pole triangles were the actual
      trigger, and it was their *combination* with a non-manifold vertex
      that destroyed the result. The pathological case went from 9.1 % to
      96.7 % retained area either way. The splitter is kept as cheap
      insurance, not as a measured necessity.
- [x] Failure policy: `min_area_retained` refuses a half-remeshed model
      instead of returning one, since AutoRemesher reports success anyway.
      `QuadRemeshStats` surfaces quad ratio, boundary edges, retained area,
      split vertices and dropped faces.
- [x] `t2glb::prepare_quad_mesh(...)` — component filter → quad remesh →
      triangulation → per-vertex PBR projection.
- [x] `mesh2glb --quad [target_quads]`, combinable with `--print`. The
      projected bake now runs for either, which is what first makes Phase
      2's tinybvh path reachable from a CLI.
- [ ] Progress bridge: `remesh()` accepts a `ProgressFn` but nothing calls
      it yet — there is no long-running caller until the server lands.
- [ ] Re-measure on a real 512³ mesh. Every number so far is synthetic, so
      the default `target_quads`, `min_area_retained` and whether
      `input_triangle_budget` should default on all remain unsettled.

### Phase 4 — C-API, server, viewer

Results: `docs/progress/autoremesher-quad-remesh_phase4-capi-server-viewer.md`.

- [x] `t2_quad_remesh_available()`, `t2_prepare_quad_mesh()` and
      `t2_quad_mesh_stats()` in `trellis2_capi.{h,cpp}`.
- [x] **`T2_CAPI_ABI_VERSION` 11 → 12**, propagated to `const abiVersion`
      in `server/engine.go`. Verified against the built DLL: the ABI check
      passes and all three symbols resolve.
- [x] Server: `quad` / `quads` / `adaptivity` job parameters, quad stage
      after the optional wrap, quad parameters in the prepare and GLB
      cache keys, `quad_remesh` in `/api/info`.
- [x] Fixed a stale gate found on the way: `BakeProjectedGLB` refused to
      run without CGAL, an assumption Phase 2 had already invalidated.
- [x] Viewer: quad checkbox with target-quads and adaptivity sliders,
      disabled with a reason when the backend is absent; quality readout
      fed by new `X-Quad-*` headers; export hints for all four
      wrap/quad combinations, including "not printable" for the pair.
- [ ] Bridge `t2quad::ProgressFn` to the server progress callback. On the
      coarse mesh the remesh is fast enough that nothing stalls; on 512³
      it will look like a hang.
- [ ] Confirm a long quad remesh cannot block progress polling — untested,
      since no slow enough run has been made through the server yet.

**First measurement on real pipeline geometry** (coarse 64³ path):
221 668 tris → 5 748 tris, 2104 quads / 347 tris / 695 n-gons,
**796 boundary edges**, 87.2 % area retained. Every synthetic fixture so
far was closed at 98 %+, so the real output is both open and lossier than
the fixtures suggested. The stage is still useful, but the UI must keep
showing these numbers instead of implying a clean retopology.

### Phase 5 — Validation and documentation

Results: `docs/progress/autoremesher-quad-remesh_phase5-validation-docs.md`.

- [x] `tests/test_quad_remesh.cpp` (ctest `quad_remesh`, no model assets,
      skips with 77 without the backend): cube/sphere fixtures,
      deterministic-output check (two runs byte-identical, statistics
      included — it holds), quad-ratio floor, index-range and finiteness
      invariants, and an explicit test that documents that boundary edges
      *may* exist. Two additions beyond the plan: the reported statistics
      are recounted from the face stream and compared, since the server
      and viewer act on them, and `triangulate()` is checked for covering
      every face exactly once.
- [x] Benchmark table in `docs/architecture/quad-remesh.md`: the Phase 1
      synthetic numbers, the Phase 4 coarse measurement, and this
      branch's 1024³ runs, plus the Phase 1 input-defect table.
- [x] `AGENTS.md`: third-party list, a new Eigen/MPL-2.0 exception bullet
      beside the CGAL one, `TRELLIS2_AUTOREMESHER` and
      `TRELLIS2_FORCE_TINYBVH` in the options table, repository tree, and
      the export section extended to `quad_remesh`.
- [x] AutoRemesher, isotropicremesher and Eigen moved out of the *Planned
      additions* table of `THIRD_PARTY.md` into the real sections;
      `THIRD_PARTY.md` linked from the `## License` section of
      `README.md`. (tinybvh was already promoted in Phase 2.)
- [x] `docs/progress/autoremesher-quad-remesh_phase5-validation-docs.md`,
      `Commits: <to be added after review>`.

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
4. **Eigen availability, not Eigen license.** With D2 the license
   question is settled (external, documented in `THIRD_PARTY.md`). What
   remains is that the `5.0...<6` floor excludes distro packages, so a
   clone without vcpkg builds without the quad stage. The
   `message(STATUS ...)` line and `t2quad::available()` have to make
   that obvious rather than puzzling.
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
11. **The manifest-mode switch touches an existing, working build.**
    Adding `vcpkg.json` changes dependency resolution for *every* vcpkg
    configure, not just the new feature: the classic
    `C:/vcpkg/installed` tree stops being consulted. This is the only
    part of the plan that can regress something that works today, which
    is why the `-DVCPKG_MANIFEST_FEATURES=cgal` edit is required in the
    same commit and verified explicitly in Phase 1.

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
