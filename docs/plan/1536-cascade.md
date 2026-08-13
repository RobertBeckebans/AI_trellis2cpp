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

- [ ] `T2_PIPE_1536` produces a 1536³ dual-grid mesh on the 32 GB card
      with no environment variables, and the run is recorded with
      measured decode VRAM, wall clock and voxel count in
      `docs/progress/`.
- [ ] The token-reduction loop reproduces upstream exactly, including
      the truncating quantization (D2) and the 1024 floor: given the
      same LR SLAT, the selected `hr_resolution` and the resulting
      coordinate set match the reference.
- [ ] The reduction is **visible**, not silent: when the loop drops
      below the requested resolution the achieved resolution reaches the
      caller (progress/stage payload or result field), mirroring
      upstream's printed notice.
- [ ] `test_cascade` gains a 1536 case, gated the same way the existing
      1024³ decode gate is (`TRELLIS2_CASCADE_DECODE`) if its footprint
      requires it.
- [ ] `T2_PIPE_1024` behaviour is bit-identical to today — every
      existing row in `docs/VERIFICATION.md` unchanged.
- [ ] `T2_CAPI_ABI_VERSION` bumped and propagated to
      `server/engine.go:20`; verified against the built DLL.
- [ ] `ctest --test-dir build -C Release -LE model` green. A `SKIP`
      (77) is not a pass.

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

**D5 — Host RAM is the unbudgeted axis.** ~9M voxels means a
correspondingly large `t2_mesh_result` (positions, normals, triangles,
six PBR channels) plus whatever the export, hole fill, orientation and
remesh stages copy. VRAM is no longer the constraint; host RAM has never
been measured for this tier.

**D6 — ABI number is claimed at merge, not now.** Both this plan and
`docs/plan/pixal3d-proj-conditioning.md` bump `T2_CAPI_ABI_VERSION` from
15. Whichever lands first takes 16; the second rebases onto 17.

## Plan (phases)

- [ ] **Phase 1 — Tier plumbing.** `T2_PIPE_1536`, `T2_CAP_1536`, the
      texture-flow selection (`src/trellis2_capi.cpp:366`), `hr_grid`
      from the requested resolution instead of `shp.resolution * 2`.
      `T2_PIPE_1024` must remain byte-identical; verify by rerunning
      `test_cascade` before touching anything else.
- [ ] **Phase 2 — Token-reduction loop.** Implemented per D2/D3, with
      the achieved resolution reported outward. Validated against a
      reference dump of `hr_resolution` and the coordinate set for at
      least one dense and one sparse object.
- [ ] **Phase 3 — Measure.** One real 1536 generation: decode VRAM peak,
      voxel count, host RAM high-water, wall clock, and how often the
      loop actually reduces. This is what fills `decode_vram_peak` (D4)
      and answers whether `max_num_tokens` should move on a 32 GB card.
- [ ] **Phase 4 — Verification and surfacing.** `test_cascade` 1536
      case, `docs/VERIFICATION.md` rows, `docs/PLAN.md:178` closed, tier
      selectable in `server/engine.go` and the viewer, ABI bump (D6).

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
- **The extrapolated 15 GB** (D4) may be wrong in either direction;
  the surface-scaling assumption ignores that the loop caps the input
  token count but not the decoder's output voxel count.
- **Host RAM** (D5) is entirely unknown for this tier.
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
