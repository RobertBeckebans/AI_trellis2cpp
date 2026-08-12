# Progress — autoremesher-quad-remesh, phase 5: validation and documentation

- **Issue:** none — tracked by plan key
- **Plan:** [`docs/plan/autoremesher-quad-remesh.md`](../plan/autoremesher-quad-remesh.md) — Phase 5
- **Branch:** `autoremesher`
- **Date:** 2026-08-12
- **Commits:** <to be added after review>

## Goal of the phase

Close the plan out so the branch can go back to `main`: an automated test for
the quad stage, the measurements written down where they can be found again,
and the license/build documentation brought in line with what actually shipped.

## What was done

**`tests/test_quad_remesh.cpp`** (ctest `quad_remesh`, no model assets, skips
with 77 in a build without the backend). It asserts structure, not output —
AutoRemesher's frame field is free to place singularities where it likes, and
pinning the result would produce a test that fails on every upstream bump for
no reason. What is checked:

- `face_sizes` exactly partitions the flat index stream, no face under three
  corners, no face repeating a corner;
- every index in range, every coordinate finite;
- the reported statistics describe *this* mesh — quad/triangle/n-gon counts and
  `boundary_edges` are recounted from the stream and compared, because those
  numbers are what the viewer and the export hints are read off;
- a 50 % quad-ratio floor, since quad-dominance is the whole point of the stage;
- `triangulate()` covers every face exactly once (`sum(n-2)` triangles), stays
  in range, and emits nothing degenerate;
- **determinism**: two runs on the sphere are byte-identical in vertices, faces,
  sizes *and* statistics;
- an empty mesh is refused rather than remeshed into nonsense.

It deliberately does **not** assert closedness. Boundary edges are a documented
legitimate outcome of the quad extraction, and the test says so in a comment
rather than leaving the next reader to guess whether their absence was luck.

**`docs/architecture/quad-remesh.md`** — how the stage relates to the Alpha Wrap
print path, the Phase 1 synthetic benchmarks, the real-pipeline numbers from
Phase 4 and from this branch's 1024³ runs, why `--target-quads` is a hint and
not a count, and the Phase 1 defect table showing that the non-manifold
*junction* is the thing that silently halves the retained area.

**`AGENTS.md`** — vendored list extended (tinybvh, autoremesher,
isotropicremesher), a new **Eigen / MPL-2.0 exception** bullet next to the
existing CGAL one, `TRELLIS2_AUTOREMESHER` and `TRELLIS2_FORCE_TINYBVH` in the
CMake options table, the repository tree corrected, and the export section
extended to cover `quad_remesh` with the explicit note that it does not replace
the wrap.

**`THIRD_PARTY.md` / `README.md`** — AutoRemesher and isotropicremesher are now
real §1 entries; Eigen is no longer "planned". The *Planned additions* table is
empty and says so, keeping the convention documented. `README.md`'s License
section now links `THIRD_PARTY.md` as the authoritative list and states both
optional-dependency obligations (CGAL/GPL, Eigen/MPL-2.0).

## Affected files

- `tests/test_quad_remesh.cpp` — new
- `tests/CMakeLists.txt` — target plus ctest registration with `SKIP_RETURN_CODE 77`
- `docs/architecture/quad-remesh.md` — new
- `docs/architecture/README.md` — Overview entry for the above
- `AGENTS.md` — third-party list, MPL2 decision, CMake options, tree, export section
- `THIRD_PARTY.md` — AutoRemesher + isotropicremesher promoted, Eigen de-planned
- `README.md` — License section, repository table
- `docs/plan/autoremesher-quad-remesh.md` — Phase 5 checked off

## Deviations / fixed along the way

1. **The test checks two things the plan did not list.** Statistics-versus-mesh
   consistency, and `triangulate()` coverage. Both were cheap and both guard
   something a caller acts on: the server puts `boundary_edges` in a response
   header and the viewer turns it into an export hint, so a statistic that
   drifts from the mesh is a wrong answer with no other detector.
2. **tinybvh was already promoted.** The plan's Phase 5 item names it, but
   Phase 2 had already moved it into §1 of `THIRD_PARTY.md`. Only AutoRemesher,
   isotropicremesher and Eigen actually moved.
3. **`docs/architecture/README.md` was still unfilled template.** Its Overview
   listed `<MODULE>.md` placeholders. Replaced those with the real entry rather
   than adding a link beside dangling angle brackets. The rest of the file is
   still boilerplate and is not this phase's business.
4. **`README.md` was stale beyond the License section.** The repository table
   listed `third_party/` as "xatlas, meshoptimizer, stb", missing tinybvh and
   autoremesher, and had no row for `src/quad_remesh.{h,cpp}`. Corrected while
   in the file.
5. **`docs/VERIFICATION.md` untouched**, as the plan requires: no neural stage
   is affected and the quad stage has no PyTorch reference to be parity-checked
   against.
6. **`format_code.bat` was not run.** This machine has clang-format 21.1.8; the
   project mandates 18.1.8, and reformatting with the wrong version would churn
   files far beyond this change. New code was hand-matched to the surrounding
   style. **This is left for the reviewer to run with the correct version.**
7. **Two fixes from the same session precede this phase on the branch** and are
   not part of it: the dual-grid winding orientation (`db2e9bd`, `61d649c`) and
   the Alpha Wrap boundary capping (`90e6ebc`). The latter is what makes the
   "after capping" row in the benchmark table possible. Both are already
   committed, so this phase's working tree holds only the files listed above.

## Verification

| Check | Result |
|---|---|
| Build `trellis2` + `test_quad_remesh` (Ninja Multi-Config, Release, clang/ROCm) | success, no new warnings |
| `test_quad_remesh` standalone | **PASS** — cube 1 344 faces / 99.4 % quads / 0 boundary / 98.6 % area; sphere 1 398 faces / 100.0 % quads / 0 boundary / 99.3 % area |
| Determinism assertion | holds — two sphere runs byte-identical including statistics |
| `ctest -LE model` (8 tests) | **8/8 passed**, `quad_remesh` 2.55 s; `preprocess` skipped (missing fixture, pre-existing) |
| `ctest -R "print_remesh\|mesh_export\|marching_cubes\|pbr_sampling"` | 4/4 passed |
| `format_code.bat` | **not run** — see deviation 6 |

No `SKIP` is counted as a pass above: `quad_remesh` genuinely ran, because this
build has the AutoRemesher backend.

## Open points

- **Phase 4 still has two unchecked items** that this phase does not close:
  bridging `t2quad::ProgressFn` to the server progress callback, and confirming
  that a long quad remesh cannot block progress polling. On a 1024³ model the
  remesh runs 70–120 s, so the missing progress bridge is now a visible gap,
  not a theoretical one.
- **The upstream AutoRemesher commit is still unpinned**
  (`third_party/autoremesher/VERSION.md`) — imported from a local source drop.
  Worth resolving before the merge, since the vendored subset cannot otherwise
  be reproduced.
- **Two benchmark rows have no wall time** (`—` in the table): the coarse 64³
  measurement from Phase 4 did not record one, and the post-capping run was
  measured for geometry rather than time.
- `docker/Dockerfile.demo` still has no Eigen 5.x; those containers report the
  stage unavailable.
- The other Windows batch scripts still do not pass the vcpkg toolchain file,
  so they build without the quad stage.

## Next phase

None — this closes the plan. The branch is ready to merge back once the
reviewer has run the formatter and decided on the two Phase 4 leftovers.

## Proposed commit message

```text
Test and document the quad remesh stage
```
