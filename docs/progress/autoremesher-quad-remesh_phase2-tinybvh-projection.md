# Progress — autoremesher-quad-remesh, phase 2: tinybvh closest-surface projection

- **Issue:** none — tracked by plan key
- **Plan:** [`docs/plan/autoremesher-quad-remesh.md`](../plan/autoremesher-quad-remesh.md) — Phase 2
- **Branch:** `autoremesher-quad-remesh` / optional
- **Date:** 2026-08-11
- **Commits:** <to be added after review>

## Goal of the phase

Take the closest-surface PBR projection off CGAL, so the texture half of the
export path works in every build instead of only in a GPL one. Alpha Wrap stays
CGAL-only; this is about `project_pbr` and everything downstream of it.

## What was done

**Vendored tinybvh 1.7.1** as `third_party/tinybvh/tiny_bvh.h`, unmodified,
plus `LICENSE` and `VERSION.md`. Nothing else from the upstream repository.

**Wrote the query tinybvh does not have.** tinybvh is a ray-tracing BVH: its
query surface is `Intersect` / `IsOccluded` / `Intersect256Rays`, plus
`IntersectSphere`, which is a boolean *overlap* test and not a nearest lookup.
So it replaces the *builder*, not the query. The nearest-point traversal is
ours — best-first descent with a point-AABB lower bound for pruning and a
Voronoi-region point-triangle distance at the leaves — written over the public
`BVH::bvhNode` / `BVH::primIdx` members, with the vendored header untouched.

**Restructured `project_pbr` so the two backends are comparable.** Validation,
the degenerate-triangle filter, the barycentric interpolation and the query
thread pool are now shared; only the nearest-surface lookup differs. Both
backends are compiled whenever CGAL is present and reachable through the new
`project_pbr_backend("cgal"|"tinybvh", ...)`, so the comparison runs in one
binary rather than needing two builds.

**Corrected the API's story about itself.** `t2print::projection_available()`
(always true) is now separate from `t2print::available()` (Alpha Wrap, still
CGAL). `mesh_to_projected_glb` no longer gates on CGAL, and the stale claims in
`mesh_export.h` and the `t2_bake_projected_glb` block in `trellis2_capi.h`
were fixed.

## Affected files

- `third_party/tinybvh/` — new: `tiny_bvh.h`, `LICENSE`, `VERSION.md`, and our
  `tiny_bvh_impl.cpp`
- `print_remesh.{h,cpp}` — tinybvh backend, own traversal, backend selection,
  `projection_available()` / `projection_backend()` / `project_pbr_backend()`
- `CMakeLists.txt` — `trellis2_tinybvh` target, `TRELLIS2_FORCE_TINYBVH`
- `mesh_export.{h,cpp}`, `trellis2_capi.h` — CGAL gate removed, comments fixed
- `tests/test_print_remesh.cpp` — rewritten; `tests/test_mesh_export.cpp` —
  obsolete assertion inverted; `tests/CMakeLists.txt` — test no longer CGAL-gated
- `THIRD_PARTY.md`, plan, this document

## Deviations / fixed along the way

1. **tiny_bvh.h's implementation half needs C++17** (`std::scoped_lock`) while
   the library is C++14 and `AGENTS.md` reserves C++17 for the CGAL path. Rather
   than raising the baseline, the one implementation TU lives in its own
   `trellis2_tinybvh` target compiled as C++17; `print_remesh.cpp` includes only
   the declaration half and stays C++14.
2. **`TINYBVH_NO_SIMD` is required, and is a deliberate choice.** tiny_bvh.h
   assumes a compiler defining `_MSC_VER` accepts SSE4.2/AVX intrinsics without
   target flags — true for MSVC, false for clang targeting the MSVC ABI, which
   fails to compile the wide-BVH ray traversal. The alternative, `-mavx2`, would
   raise `libtrellis2`'s CPU baseline to Haswell for the whole library. We use
   only the scalar builder and our own scalar traversal, so no SIMD path is
   called. The define is `PUBLIC` on the target so both TUs agree — a mismatch
   would be an ODR hazard, not just a performance difference.
3. **`tests/test_mesh_export.cpp` asserted the opposite of this phase's goal**:
   "CGAL-free projected bake did not report unavailable". Inverted to require a
   valid textured GLB instead. It failed the suite until fixed, which is the
   test doing its job.
4. **The stack-depth guard returns an error instead of a wrong answer.** The
   traversal stack is 128 deep, double what tinybvh's own ray traversal assumes.
   If it ever overflowed, dropping a subtree would produce a plausible but wrong
   nearest hit, so the query fails loudly instead.
5. No `T2_CAPI_ABI_VERSION` bump: the `trellis2_capi.h` change is comment-only,
   no signature or struct layout moved.

## Verification

| Check | Result |
|---|---|
| Configure + build, `-DTRELLIS2_CGAL=OFF`, vcpkg manifest | success |
| `ctest -LE model` (7 tests, no assets) | **100 % passed**, 0 failed |
| `test_print_remesh` standalone | PASS — `projection backend: tinybvh` |
| Barycentric correctness, 6 channels | PASS |
| Degenerate + duplicate + repeated-index fixture | PASS, result unperturbed |
| Entirely degenerate source | correctly rejected |
| `./format_code.sh` | clean; `third_party/` untouched |

### Cross-backend comparison: RESOLVED (2026-08-12)

Ran in the ROCm/HIP build produced by `cmake-ninja-win64-rocm-cgal-quad.bat`:

```
projection backend: cgal
backend agreement: max |tinybvh - cgal| = 9.77e-05 over 24576 samples
RESULT: PASS
```

4096 query points x 6 channels against a displaced grid. The two backends
agree, and the 1e-4 tolerance proposed below holds - though only just, which is
what the comment in the test predicts: the residual is which triangle wins a
near-tie at a shared edge.

**This also corrects the blocker recorded below.** The Boost.MPL failure is
**toolchain-specific, not universal**: it reproduces with the system LLVM clang
used for the local `build-quad` tree, but ROCm's clang compiles the same sources
without complaint. The CGAL path builds and runs fine on the project's reference
toolchain, so the "unbuildable here" note applies to one compiler, not to the
feature.

### Original note: comparison blocked (system LLVM clang)

The comparison the plan asks for is implemented and runs automatically, but it
**could not be executed**, because the CGAL build does not compile on this
toolchain:

```text
boost/mpl/aux_/include_preprocessed.hpp:33:11: fatal error:
  'boost/mpl/aux_/preprocessed/plain / or.hpp' file not found
```

Boost assembles that include path by token pasting; under clang the tokens stay
separate and the stringized path gets spaces. It fails at the *first* CGAL
include, before any project code. `BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS` and
`-fno-operator-names` only move the failure to the next such header
(`'boost / mpl / apply_wrap.hpp'`).

**This is not caused by this phase.** Compiling the unmodified
`git show HEAD:print_remesh.cpp` with the identical command fails the same way.
It also means the Phase 1 progress note claiming the CGAL build succeeded was
wrong; that document has been corrected.

Consequence: `test_print_remesh` prints `backend comparison skipped: the CGAL
PBR projection backend is unavailable` and passes on the tinybvh half alone.
The tolerance in the comparison (1e-4) is therefore a *proposal*, not a measured
bound.

## Open points

- The system-LLVM-clang Boost.MPL failure is understood but not fixed. It does
  not block anything (ROCm's clang builds CGAL fine) but it does mean a CGAL
  build cannot be produced from that compiler.
- Relative performance of the two backends is still unmeasured (build time,
  queries/s); only their agreement is.
- The traversal has not been profiled. `BuildHQ`/`BuildAVX` are unused, and
  `TINYBVH_NO_SIMD` costs some build speed — both are open optimisations, not
  known problems.
- A raycast (normal-offset) bake, which is what tinybvh really unlocks over
  nearest-point, remains a follow-up.

## Next phase

Phase 3, the `quad_remesh` module, which depends on Phase 1 and now starts from
the sharper input-preparation problem statement that Phase 1's measurements
produced.

## Proposed commit message

```text
Take the closest-surface PBR projection off CGAL with tinybvh
```
