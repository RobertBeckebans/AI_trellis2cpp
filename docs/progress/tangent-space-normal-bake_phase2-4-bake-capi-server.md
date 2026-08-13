# Progress — tangent-space-normal-bake, phases 2–4: tangents, the bake, and the server toggle

- **Issue:** none — tracked by plan key
- **Plan:** [`docs/plan/tangent-space-normal-bake.md`](../plan/tangent-space-normal-bake.md) — Phases 2, 3, 4
- **Branch:** `tangent-space-normal-bake` / optional
- **Date:** 2026-08-13
- **Commits:** <to be added after review>

## Goal of these phases

Turn the shading normals Phase 1 made available into an actual
tangent-space normal map in the exported GLB, and put the toggle for it in
the server UI. Phases 2 and 3 landed together because both rewrite the same
hand-assembled glTF JSON, and doing that renumbering twice would have been
the riskiest part of the work for no benefit.

## Phase 2 — tangents

**Vendored `third_party/meshoptimizer/tangentspace.cpp`** from the `v1.2`
tag, unmodified. meshoptimizer 1.2's MikkTSpace-compatible generator was
already declared in the vendored header but its translation unit had never
been imported. It turned out self-contained: `meshoptimizer.h`, `assert.h`,
`math.h`, `string.h`, and `meshopt_Allocator` from the already-vendored
`allocator.cpp`. The vendored header was verified bit-identical to upstream
`v1.2` (modulo the CRLF the whole vendored tree uses), so the subset stays
consistent.

**`build_atlas_tangents` in `mesh_export.cpp`**, between the xatlas unwrap
and the rasterizer. `meshopt_generateTangents` emits one tangent per
*corner*, so the atlas mesh is deindexed, remapped with
`meshopt_generateVertexRemapMulti` over all of position/normal/uv/tangent
(plus material when present), and reindexed. xatlas already splits at chart
boundaries, which covers UV mirror seams; this catches the in-chart
remainder. `meshopt_TangentCompatible` is set, because Blender's glTF
importer recomputes tangents with MikkTSpace instead of reading `TANGENT` —
a map baked against a differently-weighted basis would pick up shading
error in exactly the tool this feature exists to remove from the workflow.

**The generator is fed normalized UVs**, the ones the GLB carries. Texel
space would scale u and v by AW and AH separately and skew the basis
whenever the packed atlas is not square. This is why `guv` is now built
right after the unwrap instead of at write time.

## Phase 3 — the bake

**`write_glb` stopped hardcoding accessor and bufferView indices.** They
were literals, which cannot survive an attribute becoming conditional. They
are now counters handed out as streams are appended, so `TANGENT` and the
normal image can be present or absent without any index drifting. This was
flagged in the plan as the likeliest source of a silently malformed GLB and
is the change that removes that risk rather than navigating around it.

**The transfer.** The rasterizer records the covering triangle and two
barycentric weights per covered texel (12 B) rather than a materialized
frame (36 B), and the fill loop rebuilds `N`, `T`, `B = cross(N,T)*w` from
them, Gram-Schmidts `T` against `N`, and stores
`(dot(n,T), dot(n,B), dot(n,N))`. The Y-up swizzle applied at write time is
a proper rotation, so those dot products are invariant under it and the
whole transfer happens in TRELLIS space.

**The back-facing guard (D3).** Where `dot(n_source, N_target) <= 0` the
nearest source surface faces away from the texel, which on thin geometry
means the search crossed to the far side of a wall. Baking that inverts the
shading lobe over a visible patch, so the texel is left flat and counted.
Flat says "no detail here"; inverted looks like a feature until the model
is lit.

**Dilation extended to 9 channels** with renormalization — an average of
unit vectors is not one, and a non-unit gutter normal bleeds a wrong
magnitude across the seam under bilinear filtering.

**Gating.** `MeshExportOptions::normal_map` (default true) is honoured only
by `mesh_to_projected_glb`; the plain atlas bake ignores it because target
and source are the same mesh there and the map would be flat by
construction. `T2GLB_NO_NORMALMAP` is the env escape hatch.
`mesh2glb --no-normal-map` is the CLI one.

## Phase 4 — C API, server, viewer

- `t2_bake_projected_glb` gained an `int normal_map` parameter and
  `t2_last_normal_map_stats` was added → **`T2_CAPI_ABI_VERSION` 14 → 15**,
  propagated to `const abiVersion` in `server/engine.go`. Verified against
  the built DLL: all three symbols export and the server's ABI handshake
  passes at startup.
- Server: `normalmap` job parameter (default on, only an explicit
  `0`/`false` disables), threaded into `BakeProjectedGLB`, `X-Normal-Map-*`
  response headers, and a log line with the rejected share.
- The stats are published together with the cached GLB bytes
  (`j.glb, j.glbKey, j.normalMapStats` in one assignment) because the cache
  key encodes whether a map was baked; splitting them would leave stale
  numbers behind after a bake under a different key.
- Viewer: a **default-on checkbox**, disabled with a reason when neither
  replacement stage is active (without replacement geometry there is
  nothing to transfer detail onto), plus a readout of the rejected share
  from the preflight headers.

## Verification

**Round-trip test** (`tests/test_mesh_export.cpp`): a flat target quad baked
against a *tilted* flat source plane. Both frames are then constant, so
every texel must reconstruct to the same world normal and that normal must
be the source's. The test reads `NORMAL`, `TANGENT` and the normal PNG back
out of the produced GLB, decodes with stb_image, and reconstructs
`normalize(n.x*T + n.y*B + n.z*N)`.

| Check | Result |
|---|---|
| Texels reconstructed | **13456 of 13456** |
| Worst deviation from the source normal | **0.248 deg** |
| Rejected texels (source faces the target everywhere) | **0** |
| `normal_map = false` | emits neither `normalTexture` nor `TANGENT` |

0.248 deg is the 8-bit quantization floor, so handedness, green-channel
direction, the Y-up swizzle and the tangent basis are all mutually
consistent. This is the check that would have caught any of those four
being individually plausible but wrong.

Also verified: `ctest -C Release -LE model` 8/8 pass; `go build` clean;
`go vet` reports the same five pre-existing `unsafe.Pointer` purego
findings as the unmodified tree and no new ones; the server starts and
completes the ABI handshake against the rebuilt DLL.

## Confirmed on real geometry

Reported by the reviewer, 2026-08-13: the bake produces correct normals on
real generated assets on **both** replacement paths — with and without the
quad remesher — and **Blender displays the map automatically after import**,
with no manual node setup.

Two things follow from that second half. Blender's glTF importer recomputes
tangents with MikkTSpace rather than reading the exported `TANGENT`, so a
map baked against any other weighting would have shown seam and shading
error precisely there. It did not, which is the practical confirmation of
D2 and closes plan risk 2. And since the two paths hand the bake very
different target topologies — AutoRemesher quads versus an Alpha Wrap
triangle shell — the tangent basis demonstrably does not depend on the
replacement stage that produced the geometry.

## Not done

- **No quantitative measurement on real pipeline geometry.** Correctness is
  confirmed above, but the numbers are not: the rejected-texel share, the
  GLB size delta against a bake without the map, and the bake cost relative
  to the rest of the export are all still unmeasured. That also leaves plan
  risk 3 (a max-distance guard beside the back-facing one) undecided, as
  agreed — it should be decided from a measured rejection profile, not from
  a guessed threshold.
  (An earlier attempt was abandoned for picking a 208 MB full 1024³ dual
  grid, where the AutoRemesher stage rather than the bake dominates the
  runtime.)
- **The viewer cannot render the map.** It is a hand-written WebGL2
  renderer, not three.js `GLTFLoader`: no `TANGENT` attribute, no normal
  sampler. The checkbox is labelled "download only" for that reason. Adding
  it is a contained change (one attribute, one sampler, perturb N) but
  needs the preview path to carry the tangent stream and the map, so it is
  its own piece of work.
- Phase 5 (benchmark table, `AGENTS.md`, visual Blender comparison).

## Formatting

`format_code.bat` was **not** run: this machine has clang-format 21.1.8
while the project mandates 18.1.8, so running it would reformat unrelated
code. New code follows the surrounding style by hand; a reviewer with the
pinned version should run the formatter before committing.
