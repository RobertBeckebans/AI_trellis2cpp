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

## Measured on real geometry

`mesh2glb --print` at a 1024 px atlas over a textured 2.1 M-triangle
generation (1042824 verts after the component filter):

| | |
|---|---|
| Tangent split | 25959 atlas verts → **26303** over 121206 corners (**+1.3 %**) |
| Covered texels | 802165 |
| Closest-surface projection *including* normals | **0.25 s** |
| xatlas unwrap | 8.75 s |
| Texel fill / encode | 0.20 s / 0.55 s |
| Normal map texels left flat by the guard | **47906 (5.97 %)** |
| GLB | 6.36 MB, atlas 1091×1124 |

Two things worth keeping. The tangent split costs almost nothing, because
xatlas' chart boundaries already absorb nearly all of the disagreement.
And the bake is not the expensive part of the export — the unwrap is,
by a factor of 35 over the projection that now carries nine channels
instead of six.

**The back-facing guard fires on real input.** 5.97 % of covered texels
land on a source surface facing away from the low-poly texel and are left
flat. That is the D3 case actually occurring, not a theoretical one, and it
is the number the viewer readout exists to show.

## Viewer: the map is rendered, and an earlier assessment was wrong

The first pass through this recorded that the built-in viewer could not
display the map and would need the preview transport to carry a tangent
stream. That was wrong. The viewer already parses the baked GLB
(`parseGLB` / `setTexturedGLB`) and renders that exact file as the export
preview, so the only missing pieces were a `TANGENT` attribute at location
5, a third sampler, and the inverse of the bake in the fragment shader:

```glsl
vec3 T = normalize(vT);                 // the GLB's own TANGENT
vec3 B = cross(N, T) * vTw;
vec3 n = texture(uNormTex, vUV).xyz * 2.0 - 1.0;
N = normalize(T * n.x + B * n.y + N * n.z);
```

The cross product uses the *already camera-flipped* `N` the viewer applies
to its unoriented dual-grid geometry, which keeps the basis consistent on a
back-facing triangle instead of mirroring the detail there.
`colorSpaceConversion: "none"` on the `createImageBitmap` matters more for
this texture than for the other two: its bytes are a direction, and any
sRGB decode would bend it. Unit 2 always carries a texture — a 1×1
(128,128,255) when the GLB has no map — because a declared but unbound
sampler is undefined behaviour.

## Correction: no Gram-Schmidt, on either side

A first revision re-orthogonalized the tangent against the normal —
`T -= dot(T,N)*N` — in **both** the bake and the viewer shader. Raised in
review, and right: glTF defines the frame as `N`, the interpolated
`TANGENT`, and `bitangent = cross(normal, tangent.xyz) * tangent.w`, with
no such step. Doing it in both places made the bake and our own viewer
agree with each other while both drifted away from Blender and three.js —
the wrong pair to be consistent with, and a quiet reversal of the entire
reason `meshopt_TangentCompatible` was chosen. Removed from
`mesh_export.cpp` and from the fragment shader; the tangents now go from
meshoptimizer through the GLB to the shader untouched.

**The round-trip test cannot see this change** and returned an identical
0.248 deg. Its target is a flat quad, where the tangent is already exactly
perpendicular to the normal and the Gram-Schmidt was a no-op. So the
correction rests on the specification, not on that number, and the fixture
has a blind spot for frame-convention errors of this class. Closing it
needs a *curved* target, which breaks the test's constant-frame assumption
and would mean reconstructing the frame per texel — i.e. replicating the
rasterizer in the test.

**A second correction.** The preview's GLB path was gated on `printwrap`
alone, and this was first described as leaving quad remeshing without a
normal map. That overstated it: the two stages *chain* — the wrap runs
first and hands the quad remesher its clean 2-manifold — so the
wrap+quad combination always went through the GLB preview. Only quad
*without* a wrap fell back to the per-vertex path. The condition is now
`printwrap || quadremesh`, which closes that case too, and `/api/glb`
gained the `X-Quad-*` headers so previewing a quad remesh through the baked
GLB keeps its boundary-edge and retained-area readout.

Verified by replaying the viewer's parse against the produced file:
`TANGENT` as accessor 3, `normalTexture` at material level, 3 images /
3 textures / 8 bufferViews, 26303 tangents for 26303 vertices, every `w`
exactly ±1, xyz unit to 1.5e-7, and a 1091×1124 RGB8 PNG with no sRGB
chunk. The GLSL itself is only verified by review — it compiles in the
browser, not in CI.

## Not done

- **No GLB size delta** against a bake without the map (the comparison run
  was not made).
- Plan risk 3 — a max-distance guard beside the back-facing one — remains
  open, but now has a measured rejection share (5.97 %) to be decided
  against rather than a guess.
- A deterministic thin-wall fixture. The guard is observed firing on real
  input but nothing in the test suite trips it, so it is not
  regression-guarded.
- A round-trip fixture with a **curved** target, which is what would make
  the test sensitive to the tangent-frame convention it currently cannot
  distinguish (see the correction above).
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
