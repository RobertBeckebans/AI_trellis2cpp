# Progress — 1536-cascade, phases 1, 2 and 4: the third fine tier

- **Issue:** none — tracked by plan key (`1536-cascade`)
- **Plan:** [`docs/plan/1536-cascade.md`](../plan/1536-cascade.md) — phases 1, 2, 4
  (phase 3, the measurement, is **not** part of this entry)
- **Branch:** `resolution-1536`
- **Date:** 2026-08-13
- **Commits:** <to be added after review>

## Goal of the phase

Add `T2_PIPE_1536` as a fourth pipeline tier: a 1536³ dual grid from the
existing checkpoints, with upstream's token-reduction loop stepping the
resolution back toward 1024 when an object's scaffold is too dense — without
moving the 1024 tier by a single voxel.

## What was done

### The token budget, as a shared header (`src/cascade_tokens.h`)

The quantization the cascade branch used to do inline moved into a header-only
`t2cascade` namespace, plus the loop that was missing:

- `quantize_scaffold()` — the truncating `(c + 0.5) / lr_res * (res / 16)`
  formula (D2). Deliberately TRELLIS.2's form, not Pixal3D's `round()` with a
  `grid_res - 1` factor; that difference belongs to the projection-conditioning
  plan and is called out in the header comment so nobody "fixes" one into the
  other.
- `select_scaffold()` — quantize, dedup, count; while the count reaches
  `max_num_tokens` and the resolution is above 1024, subtract 128 and retry
  (D3). It returns the achieved resolution, grid, token count, and how many
  times it had to reduce.

Header-only and free of ggml state, which is what lets the pipeline and the
unit test share one implementation instead of two that can drift.

The floor test is `<=` rather than upstream's `== 1024`. Identical for both
shipped tiers, but it also terminates if a caller ever asks below the floor.

### Tier plumbing (`src/trellis2_capi.{h,cpp}`)

- `T2_PIPE_1536 = 4`, `T2_CAP_1536 = 16`. The capability rides along with
  `T2_CAP_1024`: 1536 needs no model that 1024 does not already need.
- `is_cascade_pipeline()` replaces the three scattered `pt == T2_PIPE_1024`
  tests — texture-flow selection, the texture-stage conditioning pick, and the
  availability downgrade — so a third tier could not be added to two of them
  and forgotten in the third.
- `hr_grid` now comes from what the budget settled on rather than
  `shp.resolution * 2`; `grid = hr_grid * dhp2.upscale()` already generalized.
- `decode_vram_peak()` gained a 1536 entry, **explicitly commented as
  extrapolated** rather than measured.
- `ensure_decode_vram()` is called with the resolution the loop actually chose,
  so a 1536 request that fell back to 1024 budgets like a 1024 run.
- `T2_PIPE_AUTO` still stops at 1024. 1536 costs a much larger decode; making it
  the best-available default would change every existing auto caller's runtime
  and VRAM profile silently.

### Making a reduced run visible (plan's third acceptance criterion)

Two independent channels, because a mesh that is quietly coarser than requested
is the failure mode this tier invites:

- `T2_STAGE_UPSAMPLE` fires a second time once the budget settles, with
  `step` = achieved resolution and `total` = requested. Documented at the
  `t2_progress_fn` typedef, since it is the one stage whose step/total are not a
  step counter.
- `t2_mesh_grid_resolution()` reports the grid the geometry was extracted at
  (64/512/1024/1536 or an intermediate multiple of 128).

### Server and viewer

`abiVersion` 16, `pipe1536`/`cap1536`, the new accessor bound and carried on
`meshData.GridRes` into `job.resolution`. The quality picker gains a 1536 entry;
the status line renders the upsample notice as a resolution rather than a step
count, and appends the achieved grid on completion when it differs from the tier
that was requested. The server log says so too.

The HR stage names keep their `(1024)` label at every tier — it names the model
doing the work, which is the 1024 checkpoint at 1536 as well, and the viewer's
progress weighting and timeline ranking key off those exact strings.

### Tests

- `tests/test_cascade_tokens.cpp` (new, no assets, runs in `-LE model`): the
  formula on hand-checked cells including cases where truncation and rounding
  differ; the reduction loop against per-grid counts the test derives itself,
  probed at every boundary (`count-1`, `count`, `count+1`); the 1024 floor; and
  — the load-bearing one — the 1024 scaffold against an **independent
  re-implementation** of the pre-1536 inline code, coordinate for coordinate and
  in order.
- `tests/test_cascade.cpp`: a 1536 case on the real reference upsample coords.
  Gates the 1024 selection against the existing reference through the shipped
  code path, checks 1536 invariants, and tightens to a set comparison when the
  dump carries `hr_coords_1536`.
- `scripts/dump_cascade_reference.py` now captures `hr_coords_1536` and
  `hr_resolution_1536`. Pure coordinate work on already-computed tensors, so it
  costs no extra model runs.

## Affected files

- `src/cascade_tokens.h` — **new**: quantization + reduction loop
- `src/trellis2_capi.h` — ABI 16, `T2_PIPE_1536`, `T2_CAP_1536`,
  `t2_mesh_grid_resolution`, the upsample-progress contract
- `src/trellis2_capi.cpp` — tier plumbing, budget call, VRAM entry, result field
- `tests/test_cascade_tokens.cpp` — **new** unit test
- `tests/test_cascade.cpp` — 1536 case
- `tests/CMakeLists.txt` — new target + `cascade_tokens` registration
- `scripts/dump_cascade_reference.py` — 1536 reference capture
- `server/engine.go`, `server/main.go`, `server/idle_test.go`,
  `server/web/index.html` — tier selection and resolution surfacing
- `docs/PLAN.md`, `docs/VERIFICATION.md`, `AGENTS.md`,
  `docs/plan/1536-cascade.md` — status and parity rows

## Deviations / fixed along the way

- **`T2_MAX_NUM_TOKENS` added** (not in the plan). Phase 3 has to sweep the
  budget on the 32 GB card to answer whether `max_num_tokens` should move, and
  without this that means a rebuild per data point. Unset, it reproduces
  upstream exactly.
- **The keyframe-levels risk resolved, not just noted.** The plan flagged
  `src/trellis2_capi.cpp:1046` as needing checking. At `hr_grid = 96` the first
  shift overshoots 128, so `levels` clamps to 1 and the keyframe grid is 192³
  (~28 MB scratch per frame) rather than 128³. It works; it is opt-in via
  `T2_KEYFRAMES` and off the generation path. Documented in place rather than
  changed — capping it would need a decoder call with `upsample_times = 0`,
  which is a behaviour change for a preview nicety.
- **The 1536 reference dump was not regenerated.** The script side is ready but
  producing the dump needs the PyTorch reference container. Until then the 1536
  gate in `test_cascade` is invariants-only, which is stated in
  `docs/VERIFICATION.md` rather than implied to be a parity pass.
- **Phase 3 not attempted.** See Open points.

## Verification

Ran in a scratch `build-1536-check/` (CPU, clang, Ninja; the existing `build/`
was not touched, and `cmake-ninja-win64-rocm.bat` was not run since it wipes it):

- `ctest --test-dir build-1536-check -C Release -LE model --output-on-failure`
  → **9/9, 0 failed, 1.67 s**. `cascade_tokens` passes in 0.81 s.
  Three of the nine are **SKIPs, not passes**, all for reasons predating this
  change: `quad_remesh` (AutoRemesher off in this config), `large_rows` (no GPU
  backend compiled in), `preprocess` (no `dumps/`).
- `test_cascade_tokens` in detail: all 23 assertions pass, including
  "1024 scaffold matches the legacy inline quantization coord for coord".
- ABI verified against the built DLL: `t2_abi_version()` returns **16**,
  matching `server/engine.go`, and `t2_mesh_grid_resolution` is exported.
- `go build ./... && go test ./...` in `server/` → **ok**.
  (`go vet` reports five `possible misuse of unsafe.Pointer` — all pre-existing,
  on the purego readback idiom at untouched lines 467/584/641/646/653.)

**Not verified:** every model-labelled parity test, including `test_cascade`
itself, **SKIPPED (77)** on this machine — there is no `dumps/` directory and
`ggufs/` holds only the f16 delivery weights, not the f32 validation ones. So
the claim "1024 stays byte-identical" rests here on the independent-oracle unit
test and on code inspection, **not** on a rerun of `test_cascade` against the
PyTorch reference. That rerun is the first thing the reviewer should do on a
box that has the assets.

## Open points

- **Phase 3 (measure) is untouched.** No real 1536 generation was run: it needs
  the ROCm build, the converted weights, and the target card. Still unanswered:
  decode VRAM peak, voxel count, host RAM high-water (D5, entirely unknown),
  wall clock, and how often the loop actually reduces on real objects. Until
  then `decode_vram_peak`'s 1536 entry is D4's extrapolated 15 GB.
- **The RoPE-extrapolation risk stands.** A 96³ scaffold feeds the HR model
  coordinates up to 95 when it declares `resolution: 64`. The failure mode is
  quality loss that looks like a successful run; only the side-by-side in
  phase 3/4 will catch it.
- **Diminishing returns unmeasured.** If most real objects trip the budget and
  fall back to 1408 or 1280, the tier is mostly a slower 1024. The honest
  outcome may be not to ship it on by default — which is part of why
  `T2_PIPE_AUTO` was left at 1024.
- **Reference dump** needs regenerating in the container to turn the 1536 gate
  from invariants into parity.
- **Index state:** `src/cascade_tokens.h` and `tests/test_cascade_tokens.cpp`
  show as staged (`A`) in `git status`. No `git add` was run by the agent;
  flagging it so the reviewer is not surprised by a non-empty index.

## Next phase

Phase 3 — one real 1536 generation on the R9700, then replace the extrapolated
`decode_vram_peak` entry, decide on `max_num_tokens`, and add the side-by-side
against 1024 at the same seed.

## Proposed commit message

```text
Add the 1536 cascade tier, with a token budget that can step it back down
```
