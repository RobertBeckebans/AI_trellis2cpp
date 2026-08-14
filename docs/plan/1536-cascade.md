# `1536_cascade` — the third fine tier

- **Status:** planned
- **GitHub Issue:** none — tracked by plan key (`1536-cascade`), so
  progress entries are `docs/progress/1536-cascade_<phase>-<short-name>.md`.
- **Branch:** `1536-cascade` / optional
- **Area:** trellis2_capi / server / tests / docs
- **Date:** 2026-08-13
- **Author:** AI (prepared for review)

## Goal

TRELLIS.2 defines four pipeline types — `'512'`, `'1024'`,
`'1024_cascade'`, `'1536_cascade'`
(`docs/ideas/TRELLIS-2_rocm/trellis2/pipelines/trellis2_image_to_3d.py:1374`).
This port implements the first three. `docs/PLAN.md:178` records the
fourth as still to do, and the reason it stayed there was a 16 GB card:
the 1024³ decode already peaks at ~6.75 GB, and the next tier up looked
like it would not fit.

The target machine is now a 32 GB Radeon AI PRO R9700 (`gfx1201`), which
removes that constraint. What remains is small and entirely
pipeline-level.

**Goal:** `T2_PIPE_1536` as a fourth tier — a 1536³ dual grid from the
**same** checkpoints, with an upstream-faithful token-reduction loop that
falls back toward 1024 when an object's scaffold is too dense.

## Non-goals

- **New models.** `1536_cascade` reuses
  `slat_flow_img2shape_dit_1_3B_1024_bf16` and the 1024 conditioning
  unchanged; upstream builds `cond_1024` for every non-512 tier
  (`.../trellis2_image_to_3d.py:1402`). No download, no conversion.
- **Anything from `docs/plan/pixal3d-proj-conditioning.md`.** That plan
  originally carried the 1536 work as a tail of its Phase 5; it is moved
  here because it is a TRELLIS.2 feature and independent of projection
  conditioning. Pixal3D inherits this tier once both land.
- **Changing the flow DiT.** See D1 — nothing in the forward is
  resolution-bound.
- **Raising `max_num_tokens`.** Measured first (Phase 3), decided after.

## Why this is small

The port already generalizes over the HR grid size in every place that
matters:

- **RoPE does not know the resolution.** `rope_tables_coords()`
  (`src/trellis2.cpp:2147`) builds the sparse 3D RoPE straight from the
  integer voxel coordinates — its signature has no `resolution`
  parameter, there is no clamp and no normalization. A 96³ scaffold
  simply produces larger positions. The HR flow forward needs no change
  at all.
- **The decode grid is derived.** `grid = hr_grid * dhp2.upscale()`
  (`src/trellis2_capi.cpp:1069`) already yields 1536 for `hr_grid = 96`.
- **The flow cost is capped, not scaled.** The token-reduction loop
  bounds the HR scaffold at `max_num_tokens = 49152` regardless of tier,
  so the HR flow at 1536 costs exactly what it costs at 1024 today.

What is hardcoded to 64:

| Site | Today | Needed |
|---|---|---|
| `src/trellis2_capi.cpp:998` | `hr_grid = shp.resolution * 2` → 64 | `hr_res / 16`, chosen by the loop |
| `src/trellis2_capi.cpp:41-48` | `decode_vram_peak`, two tiers | a 1536 entry |
| `src/trellis2_capi.h:51-54` | `T2_PIPE_*` enum, no 1536 | `T2_PIPE_1536` |
| `src/trellis2_capi.h:66` | `t2_caps`, no 1536 bit | `T2_CAP_1536` |
| `src/trellis2_capi.cpp:366` | `pt == T2_PIPE_1024 ? texflow_hr : texflow` | 1536 must also pick the HR texture flow |
| `src/trellis2_capi.cpp:1046` | keyframe levels loop assumes ≤128³ | check what 96³ produces |

## Acceptance criteria

- [x] `T2_PIPE_1536` produces a 1536³ dual-grid mesh on the 32 GB card
      with no environment variables, and the run is recorded with
      measured decode VRAM, wall clock and voxel count in
      `docs/progress/`. *(`1536-cascade_phase3-measure.md`. VRAM is
      recorded as a ceiling — under 16 GB — rather than an instrumented
      peak; see that entry.)*
- [~] The token-reduction loop reproduces upstream exactly, including
      the truncating quantization (D2) and the 1024 floor: given the
      same LR SLAT, the selected `hr_resolution` and the resulting
      coordinate set match the reference. *(Loop and formula gated by
      `test_cascade_tokens`; the 1024 coordinate set matches the
      PyTorch reference. The **1536** coordinate set has no reference
      dump yet — `scripts/dump_cascade_reference.py` captures it, but
      the dump needs regenerating in the container.)*
- [x] The reduction is **visible**, not silent: when the loop drops
      below the requested resolution the achieved resolution reaches the
      caller (progress/stage payload or result field), mirroring
      upstream's printed notice. *(Second `T2_STAGE_UPSAMPLE` event plus
      `t2_mesh_grid_resolution`; surfaced in the server log, the job
      JSON and the viewer. Not yet exercised by a run that actually
      reduced — 2 of 2 stayed at 1536.)*
- [x] `test_cascade` gains a 1536 case, gated the same way the existing
      1024³ decode gate is (`TRELLIS2_CASCADE_DECODE`) if its footprint
      requires it. *(Coordinate-only, so no gate needed.)*
- [~] `T2_PIPE_1024` behaviour is bit-identical to today — every
      existing row in `docs/VERIFICATION.md` unchanged. *(Held by
      construction — the loop breaks on its first pass at 1024 — and by
      `test_cascade_tokens`' independent-oracle check. **`test_cascade`
      itself has not been rerun**: the dev box has no `dumps/` and only
      f16 weights.)*
- [x] `T2_CAPI_ABI_VERSION` bumped and propagated to
      `server/engine.go:20`; verified against the built DLL.
      *(`t2_abi_version()` → 16 from the built DLL.)*
- [~] `ctest --test-dir build -C Release -LE model` green. A `SKIP`
      (77) is not a pass. *(9/9 green in a scratch CPU build; 3 of those
      are SKIPs — `quad_remesh`, `large_rows`, `preprocess` — for
      config/asset reasons predating this change.)*

## Context / affected files

| Path | Why relevant |
|---|---|
| `src/trellis2_capi.cpp:981-1070` | the whole cascade branch: upsample → quantize → HR flow → decode |
| `src/trellis2_capi.cpp:41-48`, `:117` | `decode_vram_peak`, `ensure_decode_vram` |
| `src/trellis2_capi.h:51-66`, `:130` | pipeline enum, caps, `t2_generate` |
| `src/trellis2.cpp:2147` | `rope_tables_coords` — the reason the DiT needs nothing (D1) |
| `docs/ideas/TRELLIS-2_rocm/trellis2/pipelines/trellis2_image_to_3d.py:411-424` | the reference loop |
| `server/engine.go`, `server/web/` | tier selection reaches the UI |
| `docs/PLAN.md:178` | the roadmap entry this closes |

## Architecture decisions

**D1 — No change below the pipeline layer.** The HR flow is the same
model on more voxels. Because the sparse RoPE is built from coordinates
(`src/trellis2.cpp:2147`), a 96³ scaffold is not a special case. This
plan therefore touches `trellis2_capi` and not `trellis2.cpp` — a useful
invariant to state up front, because any pressure to modify the DiT
means something has been misunderstood.

**D2 — Quantize by truncation, matching TRELLIS.2.** The port's
`(int32_t)((c + 0.5f) / lr_res * hr_grid)` (`src/trellis2_capi.cpp:1003`)
reproduces upstream's `.int()` (`.../trellis2_image_to_3d.py:415`)
exactly. Note that **Pixal3D changed this** to
`((c + 0.5) / lr_res * (grid_res - 1)).round()`
(`docs/ideas/Pixal3D/pixal3d/pipelines/pixal3d_image_to_3d.py:717`) —
different rounding *and* a different scale factor. This plan keeps the
TRELLIS.2 form; whether the Pixal3D path needs its own is a question for
the other plan, not this one.

**D3 — The loop is a safety net, not a preference.** Start at the
requested tier, quantize, dedup, count; while the count is at or above
`max_num_tokens` and the resolution is above 1024, subtract 128 and
retry. The floor exists because 1024 is the tier that is known to fit.
Keeping the floor means a 1536 request can never be worse than a 1024
request.

**D4 — Budget from measurement, extrapolate only to plan.** Known: 512³
decode ~2.4 GB, 1024³ ~6.75 GB at ~3.97M voxels
(`src/trellis2_capi.cpp:41`, `docs/VERIFICATION.md:50`). Surface-area
scaling to 1536³ suggests ~9M voxels and ~15 GB. On 32 GB, with
`ensure_decode_vram` freeing the finished flow DiTs (5–7 GB) beforehand,
that fits with room. **This number is extrapolated and must be replaced
by a measurement in Phase 3 before `decode_vram_peak` gets its 1536
entry** — a low estimate there silently pushes the decoder to the CPU
instead of failing loudly.

*Phase 3 outcome: D4 put the risk in the wrong place.* The 1536³ decode
measured **20.2 s with the whole run under 16 GB of 32** — barely 1.6×
the 1024³ decode, not the tier's problem at all. What actually costs is
the HR flow (122.6 s, and ~4.8× that on a scaffold near the token
ceiling) and the texture stage (117.7 s). The 15 GB entry stays: the
reading is a ceiling taken while the flow DiTs were still resident, so
it rules out the dangerous direction (too low) without pinning the
transient share. Lowering it would need `trellis2_gpu_free_vram()`
sampled either side of the decode.

**D5 — Host RAM is the unbudgeted axis.** ~9M voxels means a
correspondingly large `t2_mesh_result` (positions, normals, triangles,
six PBR channels) plus whatever the export, hole fill, orientation and
remesh stages copy. VRAM is no longer the constraint; host RAM has never
been measured for this tier.

**D6 — ABI number is claimed at merge, not now.** Both this plan and
`docs/plan/pixal3d-proj-conditioning.md` bump `T2_CAPI_ABI_VERSION` from
15. Whichever lands first takes 16; the second rebases onto 17.

## Plan (phases)

- [x] **Phase 1 — Tier plumbing.** `T2_PIPE_1536`, `T2_CAP_1536`, the
      texture-flow selection (`src/trellis2_capi.cpp:366`), `hr_grid`
      from the requested resolution instead of `shp.resolution * 2`.
      `T2_PIPE_1024` must remain byte-identical; verify by rerunning
      `test_cascade` before touching anything else.
- [x] **Phase 2 — Token-reduction loop.** Implemented per D2/D3, with
      the achieved resolution reported outward. Validated against a
      reference dump of `hr_resolution` and the coordinate set for at
      least one dense and one sparse object.
      *Landed as `src/cascade_tokens.h`. The dump side is prepared
      (`scripts/dump_cascade_reference.py` now captures `hr_coords_1536`
      and `hr_resolution_1536`), but the dump itself has not been
      regenerated — that needs the reference container. Until it is,
      `test_cascade` gates the 1536 selection on invariants and the
      1024 tier on the existing reference.*
- [x] **Phase 3 — Measure.** One real 1536 generation: decode VRAM peak,
      voxel count, host RAM high-water, wall clock, and how often the
      loop actually reduces. This is what fills `decode_vram_peak` (D4)
      and answers whether `max_num_tokens` should move on a 32 GB card.
      *Done on the R9700, 2026-08-13 —
      `docs/progress/1536-cascade_phase3-measure.md`. Full 1536³ on 2 of
      2 objects, no reduction; 315 s total; the 1536³ decode is only
      20.2 s and VRAM stays under 16 GB, so **D4 had the risk in the
      wrong place** — the HR flow (122.6 s, ~4.8× that on a dense
      object) and the texture stage (117.7 s) dominate.
      **`max_num_tokens` should not move:** the ceiling costs time, not
      memory. `decode_vram_peak` deliberately keeps the conservative
      extrapolation — the reading bounds the peak but does not pin it.
      Not measured: host RAM (D5), and the RoPE-quality side-by-side.*
- [x] **Phase 4 — Verification and surfacing.** `test_cascade` 1536
      case, `docs/VERIFICATION.md` rows, `docs/PLAN.md:178` closed, tier
      selectable in `server/engine.go` and the viewer, ABI bump (D6).
      *ABI 16 taken per D6; the projection-conditioning plan rebases
      onto 17. A second unit test, `test_cascade_tokens`, covers the
      loop without assets in the `-LE model` set.*

## Tests / verification

- `test_cascade`, extended: quantized 96³ scaffold against the reference
  coordinate set, same gate as the existing 64³ check (currently exact
  to 1 voxel of ~995k).
- The token loop as a unit check on a recorded LR SLAT — deterministic,
  no GPU, belongs in the `-LE model` set.
- `ctest --test-dir build -C Release -LE model` green throughout.
- Manual smoke test: same image and seed at 1024 and 1536, side by side
  in `docs/progress/`, with the measured numbers from Phase 3.

## Risks / open questions

- **RoPE extrapolation.** The HR model declares `resolution: 64`, and a
  96³ scaffold feeds coordinates up to 95 — beyond anything it saw in
  training. Upstream sanctions this, and RoPE extrapolates by
  construction rather than failing, so the failure mode is *quality
  loss that looks like a successful run*. The side-by-side in Phase 4 is
  the only thing that will catch it.
  *Phase 3 outcome: largely retired.* A character subject at 1536
  reconstructs jacket emblem, shoulder spikes, separate zipper runs,
  goggle band and boot soles as distinct geometry — precisely the scales
  that would break up first. Reviewer's verdict versus the lower tiers:
  noticeably better. Not yet backed by a recorded same-seed
  1024-vs-1536 pair, so the claim rests on inspection rather than an
  artefact.
- **The extrapolated 15 GB** (D4) may be wrong in either direction;
  the surface-scaling assumption ignores that the loop caps the input
  token count but not the decoder's output voxel count.
  *This second sentence turned out to be the important one, and not for
  VRAM. A jester figure at 1536 drove decoder level 3 to 2,151,017
  voxels against this backend's 2,097,152 mul_mat column cap and failed
  the decode outright, after 407 s of HR flow. The same object completes
  at 1024, because the coarser 64³ quantization merges more voxels and
  level 3 inherits the smaller scaffold. So `max_num_tokens` is
  implicitly a decode-size limit and 49,152 is too generous here — see
  `docs/bugs/rocm-texture-stage-invalid-configuration.md`.*
- **Host RAM** (D5) is entirely unknown for this tier.
  *Phase 3: the generation half is now bounded — a 3.32M-vertex /
  6.66M-triangle result is ~228 MiB in `t2_mesh_result`, ~0.5 GB
  including the server's copy. The export chain (CGAL alpha wrap over
  all 6.66M triangles) is the remaining unmeasured consumer.*
- **The keyframe preview path** (`src/trellis2_capi.cpp:1046`) computes
  its level count against a ≤128³ target; at `hr_grid = 96` the first
  shift already exceeds it. Needs checking, not assuming.
- **Diminishing returns.** If most real objects trip the loop and fall
  back to 1408 or 1280, the tier is mostly a slower 1024. Phase 3 should
  report the reduction rate, and the honest outcome may be that the tier
  is not worth shipping by default.

## Release note

New 1536 cascade tier: a 1536³ dual grid from the existing checkpoints,
with an automatic fallback toward 1024 when an object's voxel scaffold
would exceed the token budget. Requires a card that can hold the larger
decode.

## Proposed commit message

```text
Add the 1536 cascade tier, with a token budget that can step it back down
```
