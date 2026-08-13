# Tangent-space normal map bake for the projected atlas

- **Status:** in progress
- **GitHub Issue:** none — tracked by plan key
  (`tangent-space-normal-bake`), so progress entries are
  `docs/progress/tangent-space-normal-bake_<phase>-<short-name>.md`.
- **Branch:** `tangent-space-normal-bake` / optional
- **Area:** print_remesh / mesh_export / trellis2_capi / third_party /
  server / docs
- **Date:** 2026-08-13
- **Author:** AI (prepared for review)

## Goal

`t2glb::mesh_to_projected_glb` already transfers the dense voxel mesh's
material onto a low-poly target's UV atlas: every covered texel is
projected to the closest source triangle by `t2print::project_pbr`
(tinybvh) and receives barycentrically interpolated base color,
metallic, roughness and alpha.

What it does **not** transfer is *shape*. The quad remesh drops from
~10^5–10^6 triangles to a few thousand, and everything the dense mesh
knew about surface detail is gone from the export. Recovering it today
means loading both meshes into Blender, building a cage, and baking a
normal map by hand — for every generation.

The closest-surface query already computes exactly what a normal bake
needs: a source triangle plus barycentric weights per texel. Reading
three more interpolated channels (the source's shading normal) off that
same hit costs no extra BVH traversal. The only missing piece is a
tangent frame on the low-poly target, which `meshopt_generateTangents`
produces from the atlas mesh's positions, normals and UVs.

**Goal:** the projected bake emits a third texture — a tangent-space
normal map — plus a `TANGENT` vertex attribute, so the downloaded GLB
carries the dense mesh's surface detail and Blender is no longer part
of the pipeline.

## Non-goals

- Replacing the closest-point query with a raycast bake. That is still
  the follow-up recorded in `docs/plan/autoremesher-quad-remesh.md`
  (D5). This plan makes its absence *visible* (see D3) rather than
  fixing it.
- Displacement, height, AO or curvature maps. The same machinery would
  carry them, but each needs its own quality argument.
- A normal map for the non-projected `mesh_to_glb` atlas. There the
  target *is* the source, so the map is flat by construction.
- Object-space normal maps. Tangent space is what the target formats
  (glTF, Blender, game engines) expect.

## Acceptance criteria

- [x] The projected bake writes a GLB with a `normalTexture` and a
      `TANGENT` accessor whose shading reproduces the dense mesh's
      detail. **Confirmed on real generated geometry (2026-08-13), on
      both replacement paths — with and without the quad remesher —
      and Blender displays the map automatically on import with no
      manual node setup.** That last part is the practical proof of D2:
      Blender recomputes tangents with MikkTSpace rather than reading
      `TANGENT`, so this only works because the bake used a
      MikkTSpace-compatible basis.
- [x] Round-trip test: a flat target against a tilted source plane,
      reconstructing `normalize(T*r + B*g + N*b)` from the baked texel
      and the exported frame — 13456/13456 texels, worst **0.248 deg**,
      the 8-bit quantization floor.
- [x] The bake reports how many texels the back-facing guard (D3) left
      flat, through `BakeTimings`, the C API, an `X-Normal-Map-*`
      header and a viewer readout.
- [ ] No second BVH traversal: `BakeTimings::projection` for a bake
      with the normal map is within noise of the same bake without it.
      (True by construction — the normal rides on the existing hit —
      but not measured.)
- [x] `T2_CAPI_ABI_VERSION` bumped 14 → 15 and propagated to
      `server/engine.go`; verified against the built DLL.
- [x] The newly vendored `tangentspace.cpp` is listed in
      `THIRD_PARTY.md`.
- [x] `ctest --test-dir build -C Release -LE model` green, 8/8. The
      pre-existing `preprocess` skip is unrelated and is not counted as
      a pass.
- [x] No tap in `docs/VERIFICATION.md` changes — this plan touches no
      neural stage.

## Context / affected files

| Path | Why relevant |
|---|---|
| `src/print_remesh.{h,cpp}` | `project_pbr` / `project_with`; gets the source-normal channels |
| `src/mesh_export.cpp` | `xatlas_unwrap`, `bake_atlas_locked`, `write_glb` — the whole bake |
| `src/mesh_export.h` | `MeshExportOptions`, `BakeTimings` |
| `src/trellis2_capi.{h,cpp}` | `t2_bake_projected_glb` gains a flag → **ABI bump** |
| `server/engine.go`, `server/web/` | ABI propagation, job parameter, viewer toggle |
| `third_party/meshoptimizer/` | one more upstream source file (see D2) |
| `examples/mesh2glb.cpp` | CLI flag |
| `tests/test_print_remesh.cpp`, `tests/test_mesh_export.cpp` | new coverage |

## Architecture decisions

### D1 — Reuse the hit, do not run a second query

`project_with` in `print_remesh.cpp` already produces, per query point,
a triangle index and clamped barycentric weights `(wa, wb, wc)`. The
six PBR channels are interpolated from a per-vertex `source_pbr`
array. The source's shading normal is another per-vertex array of the
same shape — `PreparedMesh::normals`, which `prepare_mesh_locked`
already fills.

So the change is to interpolate **9** channels instead of 6, not to add
a second traversal. Concretely:

- `project_with` takes an optional `source_normals` pointer and an
  optional `out_normals`, interpolated with the same weights, then
  normalized (barycentric interpolation of unit vectors is not unit).
- A new public entry point beside the existing pair:

  ```cpp
  // Same closest-surface search as project_pbr, but also returns the
  // source's interpolated shading normal at each hit. Pass nullptr for
  // `backend` to use the build-time default. out_normals receives three
  // floats per query, unit length; a query whose hit triangle has no
  // usable normal receives (0,0,0), which the caller must treat as
  // "no sample" rather than as a direction.
  TRELLIS2_API bool project_pbr_and_normals( const char* backend,
      const std::vector<float>&   source_verts,
      const std::vector<int32_t>& source_tris,
      const std::vector<float>&   source_pbr,
      const std::vector<float>&   source_normals,
      const std::vector<float>&   query_points,
      std::vector<float>&         out_pbr,
      std::vector<float>&         out_normals,
      std::string&                err );
  ```

  `project_pbr` and `project_pbr_backend` keep their signatures and
  become thin forwarders, so nothing downstream changes shape.

**Interpolated vertex normals, not face normals.** The dense mesh is a
dual-grid extraction whose per-face normals are blocky; its per-vertex
normals are what the vertex-colour export already shades with. Baking
the face normal would bake the voxel staircase into the map.

### D2 — Tangents from meshoptimizer's MikkTSpace implementation

meshoptimizer 1.2 ships its own MikkTSpace-compatible tangent
generator, which is both faster than the original `mikktspace.c` and
already MIT under the vendoring rules this project follows. That
settles the question — there is no reason to reach for a second
library or for an own construction.

One piece of bookkeeping was needed: `meshoptimizer.h` declares
`meshopt_generateTangents`, but the vendored subset (`allocator`,
`indexgenerator`, `simplifier`, `vfetchoptimizer`) did not include the
implementing translation unit. **Imported:**
`third_party/meshoptimizer/tangentspace.cpp` from the `v1.2` tag,
unmodified, added to the `trellis2` target. It is self-contained — it
includes only `meshoptimizer.h`, `assert.h`, `math.h` and `string.h`,
and the one meshoptimizer symbol it reaches for outside itself is
`meshopt_Allocator`, whose `allocator.cpp` was already vendored. The
vendored `meshoptimizer.h` was verified bit-identical to the `v1.2`
header (modulo the CRLF endings the whole vendored tree uses), so the
subset stays internally consistent.

Two properties shape how it is called:

1. **`meshopt_TangentCompatible` is set.** The header describes it as
   "produce tangents compatible with MikkTSpace (same weighting and
   fallbacks) at the cost of reduced quality. Not recommended unless
   normal maps are baked." That is precisely this case. It matters
   beyond taste: Blender's glTF importer **recomputes** tangents with
   MikkTSpace rather than importing `TANGENT`, so a map baked against a
   non-MikkTSpace basis develops seam and shading error the moment the
   asset is opened there — which is the exact workflow this plan is
   meant to remove.
2. **Output is per-corner** (`index_count * 4`), so the atlas mesh has
   to be split wherever two corners of a shared vertex disagree.
   xatlas already splits at every chart boundary, which covers UV
   mirror seams, but not in-chart disagreement.

Splitting uses functions already vendored in `indexgenerator.cpp`:
deindex the atlas mesh to per-corner streams (position, normal, uv,
tangent), call `meshopt_generateVertexRemapMulti` over all four, then
`meshopt_remapIndexBuffer` + `meshopt_remapVertexBuffer` per stream.
Position/normal/uv corners coming from the same xatlas vertex are
bit-identical, so the remap only ever splits on a genuine tangent
difference.

**Ordering consequence:** the tangent + reindex step must run *between*
`xatlas_unwrap` and the rasterizer, because it changes `onv` and
`oidx`. The rasterizer then interpolates the frame from the same
vertices it interpolates position from.

### D3 — The back-facing guard, because closest-point is not a bake cage

A nearest-point transfer picks the geometrically closest surface, which
on thin geometry is happily the *other side of the wall*. For a colour
channel that is a subtle error. For a normal it inverts the entire
shading lobe, and it does so in a patch large enough to see.

This plan does not fix that — the raycast bake does, and it stays a
follow-up. What this plan does is refuse to emit a confidently wrong
normal:

- if `dot(n_source, N_target) <= 0`, the hit is on a surface facing
  away from the low-poly texel and is treated as **no sample**: the
  texel gets flat `(0, 0, 1)`.
- the count of rejected texels is reported (new `BakeTimings` field, C
  API, server header, viewer readout).

Flat is honest — it says "no detail here". Inverted is a bug that looks
like a feature until someone lights the model. A fixture with a thin
wall must show a non-zero rejection count, so the number is proven to
mean something rather than being permanently zero.

### D4 — Encoding and glTF assembly

- **Texture:** RGB8 PNG, `n * 0.5 + 0.5` per channel, uncovered texels
  and rejected samples `(128, 128, 255)`. glTF defines `normalTexture`
  as linear data, and the PNG carries no `sRGB` chunk, so no colour
  management applies.
- **Material:** `"normalTexture":{"index":2,"scale":1.0}` beside the
  existing base-colour and metallic-roughness textures.
- **Attribute:** `TANGENT` as `VEC4` float (xyz + handedness w), per
  the glTF spec. `write_glb` gains one accessor and two bufferViews;
  the hardcoded accessor/bufferView indices in its JSON must be
  renumbered carefully — that block is the likeliest source of a
  silently malformed GLB.
- **Space:** positions and normals are swizzled to Y-up
  (`x, z, -y`) at write time. That swizzle is a proper rotation, so
  `dot(T, n)`, `dot(B, n)`, `dot(N, n)` are invariant under it and the
  bake can be computed in TRELLIS space. The tangent vector itself is
  swizzled identically; `w` is unaffected.
- **Handedness / V direction:** the bake derives `B = cross(N, T) * w`
  from the *exported* UVs, and the renderer reconstructs it the same
  way from the same UVs, so the convention round-trips regardless of
  whether V runs up or down the image. This must not be "fixed" by
  flipping the green channel on a hunch — the round-trip test in the
  acceptance criteria is what settles it.
- **The frame is the glTF frame, unmodified.** `N`, the interpolated
  `TANGENT` straight out of meshoptimizer, and `B = cross(N, T) * w`.
  No Gram-Schmidt re-orthogonalization on either side. It is tempting,
  since interpolation across a triangle bends `T` slightly off `N`, and
  an earlier revision did it in both the bake and the viewer — which
  made the two agree with each other while both drifted away from
  Blender and three.js. That is the wrong pair to be consistent with,
  and it quietly undoes the whole reason for D2. Anything that reads
  this file reconstructs the frame per the spec, so the bake must
  encode against exactly that frame.
- **Gutter:** the dilation pass currently averages 6 channels; it
  extends to 9, with the normal renormalized after averaging (an
  average of unit vectors is not one).

### D5 — Gating and defaults

- `MeshExportOptions` gains `bool normal_map = true`, honoured only by
  `mesh_to_projected_glb`. The plain `mesh_to_glb` atlas bake ignores
  it: target and source are the same mesh there, so the map would be
  uniformly flat and would only cost a third PNG.
- `T2GLB_NO_NORMALMAP` disables it, matching the existing
  `T2GLB_XATLAS` / `T2GLB_NOSMOOTH` escape-hatch style.
- `t2_bake_projected_glb` gains an `int normal_map` parameter →
  `T2_CAPI_ABI_VERSION` bump → `server/engine.go` propagation. This is
  binding per `CLAUDE.md`, not optional.

Default-on is deliberate: the projected bake exists *because* geometry
was replaced, so the detail transfer is the point rather than a
decoration. The cost is one more 2048² PNG in the GLB.

### D6 — Cost, stated up front

Per projected bake, on top of today's:

- one `float[3]` per query point returned from the projection
  (`+12 B/texel`, transient);
- three more `float` channels over the atlas (`+12 B/texel`, held for
  the dilation pass);
- one more RGB8 image (`AW*AH*3`) and its PNG;
- the tangent generation and remap over the atlas mesh — linear in
  atlas vertex count, negligible against xatlas' own unwrap.

No extra BVH traversal, which is the one cost that would have mattered.

## Plan (phases)

### Phase 1 — Source normals out of the projection

Results: `docs/progress/tangent-space-normal-bake_phase1-source-normals.md`.

- [x] `project_with` interpolates an optional 3-channel normal stream
      alongside the 6 PBR channels; normalize after interpolation,
      emit `(0,0,0)` when the interpolated vector has no length.
- [x] `project_pbr_and_normals` in `print_remesh.{h,cpp}` per D1; all
      three entry points forward to a shared `project_dispatch`.
- [x] `tests/test_print_remesh.cpp`: 96×48 analytic sphere fixture —
      **0.6009 deg** worst deviation for both backends (the
      tessellation's own budget), unit length to 4.27e-08.
- [x] PBR channels verified **bit-identical** with and without the
      normal request.
- [x] **Finding:** the two backends differ on ~half of all samples
      (2021/4096), not on a handful of shared-edge ties — CGAL's exact
      predicates in double vs tinybvh's float shift the barycentric
      weights across the whole field, by at most 0.023 deg. The
      existing material-comparison comment claimed ties because it only
      ever inspected the maximum; corrected in place, and the normal
      test asserts against the analytic truth instead.

Independently reviewable; nothing downstream depends on it yet.

### Phase 2 — Tangents on the atlas mesh

Results: `docs/progress/tangent-space-normal-bake_phase2-tangents.md`.

- [x] Import `third_party/meshoptimizer/tangentspace.cpp` from the
      `v1.2` tag, add it to `CMakeLists.txt`, extend the
      `THIRD_PARTY.md` entry's vendored-file list. Verified
      self-contained (no translation unit beyond the already-vendored
      `allocator.cpp`) and the vendored header verified bit-identical
      to upstream `v1.2`.
- [x] Tangent generation + `generateVertexRemapMulti` split between
      `xatlas_unwrap` and the rasterizer, with
      `meshopt_TangentCompatible` set. Fed the *normalized* UVs, since
      texel space skews the basis on a non-square atlas.
- [x] `write_glb` emits `TANGENT`. The hardcoded accessor/bufferView
      literals were replaced by counters handed out as streams are
      appended, so a conditional attribute cannot shift an index.
- [x] Vertex cost of the split, measured on a real 2.1 M-triangle
      textured mesh: **25959 atlas verts → 26303** over 121206 corners,
      **+1.3 %**. xatlas' chart splitting really does absorb almost all
      of it.

### Phase 3 — The bake

Results: `docs/progress/tangent-space-normal-bake_phase3-bake.md`.

- [x] Rasterizer records the hit triangle and two barycentric weights
      per covered texel (12 B) rather than a materialized frame, and
      reconstructs `N`, `T`, `B` in the fill loop.
- [x] Tangent-space projection, the D3 back-facing guard and its
      counter, `(0,0,1)` fallback, 9-channel dilation with
      renormalization, RGB8 PNG, `normalTexture` in the material.
- [x] `MeshExportOptions::normal_map`, `T2GLB_NO_NORMALMAP`,
      `BakeTimings` extended with the rejection count.
- [x] **Round-trip verified**: flat target against a tilted source
      plane, read back out of the produced GLB and reconstructed —
      13456/13456 texels, worst **0.248 deg** (the 8-bit quantization
      floor), 0 rejected. Handedness, green channel, swizzle and basis
      are mutually consistent.

### Phase 4 — C API, CLI, server, viewer

Results: `docs/progress/tangent-space-normal-bake_phase4-capi-server.md`.

- [x] `t2_bake_projected_glb` flag plus `t2_last_normal_map_stats`,
      **`T2_CAPI_ABI_VERSION` 14 → 15**, `server/engine.go` propagation.
      Verified against the built DLL: symbols export and the server's
      ABI handshake passes at startup.
- [x] `mesh2glb --no-normal-map` (default on).
- [x] Server job parameter (default on), GLB cache key,
      `X-Normal-Map-*` headers, log line with the rejected share. The
      stats are published in the same assignment as the cached GLB, so
      a bake under a different key cannot leave stale numbers behind.
- [x] Viewer: default-on checkbox, disabled with a reason when neither
      replacement stage is active, plus a rejected-share readout.
- [ ] `/api/info` capability flag — not added; nothing gates on it,
      since the bake needs no optional backend.

### Phase 5 — Validation and documentation

Results: `docs/progress/tangent-space-normal-bake_phase5-validation-docs.md`.

- [x] Analytic round-trip test (in `tests/test_mesh_export.cpp`).
- [ ] Thin-wall rejection *test*. The guard demonstrably fires on real
      geometry — **47906 of 802165 covered texels, 5.97 %**, on the
      2.1 M-triangle mesh above — so it is no longer merely reachable in
      principle. What is still missing is a deterministic fixture that
      trips it, so the behaviour is observed but not regression-guarded.
- [ ] Bake-cost and GLB-size table beside the quad-remesh benchmarks in
      `docs/architecture/`. First real data point: 1024 px atlas over
      2.1 M source triangles → unwrap 8.75 s, projection **0.25 s**
      *including* the shading normals, texel fill 0.20 s, encode 0.55 s,
      6.36 MB GLB. The bake is not the cost; the unwrap is.
- [ ] `AGENTS.md` third-party list and options table.
- [x] `THIRD_PARTY.md` extended with `tangentspace.cpp`.
- [x] Blender check: an exported asset imports and shades correctly
      with no manual setup, on both replacement paths (2026-08-13).
- [ ] A side-by-side picture for the docs — the behaviour is confirmed,
      but nothing in the repository shows it yet.

## Tests / verification

- `ctest --test-dir build -LE model` — new `print_remesh` normal
  coverage and the `mesh_export` round-trip run without model assets.
- Round-trip: low-poly target + analytic high-poly source →
  reconstruct the world normal from the baked texel and the exported
  frame → compare against the analytic normal. This is the test that
  catches a green-channel flip, a handedness error, or a swizzle
  mistake, all of which look plausible by eye.
- Thin-wall fixture → non-zero D3 rejection count.
- `./format_code.sh && git diff --stat` — empty diff under
  `third_party/`.
- Manual: `mesh2glb sample.t2mesh out_quad.glb --quad 20000`, opened in
  a glTF viewer **and** in Blender (which recomputes tangents — the
  case D2 exists for).
- `docs/VERIFICATION.md` untouched.

## Risks / open questions

1. **Closest-point is not a cage.** D3 makes the failure visible and
   flat instead of inverted, but the real fix is the raycast bake. On
   very thin geometry the map may end up mostly flat — that is the
   honest answer, and it is also the argument for prioritizing the
   follow-up.
2. ~~**Blender ignores `TANGENT`.**~~ **Resolved by observation
   (2026-08-13):** a baked asset imports into Blender and shows the
   normal map correctly with no manual setup, on both replacement
   paths. Since Blender recomputes the basis with MikkTSpace instead of
   reading `TANGENT`, that agreement is exactly the property
   `meshopt_TangentCompatible` was chosen for — had the basis differed,
   this is where it would have shown.

2b. ~~**The built-in viewer cannot show the map.**~~ **Resolved, and the
   original assessment was wrong.** The viewer is a hand-written WebGL2
   renderer, but it already parses the baked GLB (`parseGLB` /
   `setTexturedGLB`) and renders that exact file for the export preview —
   there was never a preview-transport problem. It needed only a `TANGENT`
   attribute at location 5, a third sampler, and the inverse of the bake in
   the fragment shader. Done; the checkbox no longer says "download only".
3. **Low-poly / high-poly divergence.** Where the quad remesh strays
   far from the source (the quad stage already measures 87 % retained
   area on real geometry), the closest point is far away and the baked
   normal is meaningless regardless of tangent correctness. A
   max-distance cutoff, expressed as a fraction of the bounding-box
   diagonal, may be needed as a second guard beside D3.
   **Deferred by decision (2026-08-13):** revisit after the Phase 3
   bake produces a real measurement, rather than guessing a threshold
   now.
4. **Atlas resolution.** Normal maps show undersampling much more than
   colour does. 2048² may prove too small for a detailed source; the
   bake should at least report texel density so the case is visible.
5. **The `write_glb` JSON is hand-assembled.** Adding an accessor and
   two bufferViews means renumbering hardcoded indices in a `snprintf`
   block. A wrong index produces a GLB that loads in one viewer and
   not another. Validate, do not eyeball.
6. ~~**The vendored file may pull in more than expected.**~~ Resolved
   at import: `tangentspace.cpp` is self-contained (see D2).
7. **GLB size.** A third full-resolution PNG on every projected
   download. Worth measuring against the existing two before defaulting
   it on for the server path.

## Follow-ups (own plans, not in scope here)

- **Raycast bake with a cage**, replacing the nearest-point transfer
  (already recorded in `docs/plan/autoremesher-quad-remesh.md`).
- **Ambient-occlusion and curvature bakes** over the same hit data.
- **Quad-preserving container** so the exported asset keeps its quads.

## Release note

The projected GLB bake now transfers surface detail as well as
material: it emits a MikkTSpace-compatible tangent attribute and a
tangent-space normal map sampled from the dense voxel mesh, so a
quad-remeshed export no longer has to be normal-baked by hand.

## Proposed commit message

Phase 1 (the first reviewable unit) would be committed as:

```text
Return the source shading normal from the closest-surface projection
```
