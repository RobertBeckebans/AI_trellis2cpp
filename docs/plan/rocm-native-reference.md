# `rocm-native-reference` — produce the PyTorch reference dumps on the R9700

- **Status:** in progress — phases 0-2 done, phase 3 open
- **GitHub Issue:** none — tracked by plan key (`rocm-native-reference`), so
  progress entries are `docs/progress/rocm-native-reference_<phase>-<name>.md`.
- **Branch:** `resolution-1536` (current) or a fresh one
- **Area:** scripts / tests / docs
- **Date:** 2026-08-14
- **Author:** AI (prepared for review)

## Goal

Make the PyTorch reference dumps producible on the development machine, so the
parity suite stops being dark, and close the two `[~]` acceptance criteria left
open by `docs/plan/1536-cascade.md`.

## Why this is worth a plan

**The parity suite currently verifies nothing on this box.** `models/`,
`dumps/` and `tests/ss_sample_ref.bin` are all absent, so `test_dino`,
`test_slat`, `test_cascade` and `test_texture` every one exit 77. That is not a
1536 problem — it predates the tier — but the 1536 work is what made it
visible, and two of its acceptance criteria hang on it:

- criterion 2: the token-reduction loop matches upstream's coordinate set at
  1536 (currently invariants only, no reference to compare against)
- criterion 5: `T2_PIPE_1024` is bit-identical (currently held by an
  independent-oracle unit test and by the shape of the loop, not by the
  PyTorch reference)

**The container path cannot use this machine's GPU.** `docker/Dockerfile.ref`
is `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel`; the card is a Radeon AI PRO
R9700. Everything would fall back to CPU, and
`scripts/dump_cascade_reference.py` runs a 12-step sampler with the 1024 model
over ~40k tokens — minutes on a GPU, hours on a CPU.

**A native ROCm path exists.** The reviewer runs PyTorch on this card through
`uv` with AMD's Windows ROCm wheels (`repo.radeon.com/rocm/windows/rocm-rel-7.2`,
torch 2.9.1+rocmsdk, CPython 3.12). ROCm PyTorch presents itself as device
`cuda`, so the scripts' `--device cuda` needs no change.

## Non-goals

- **Changing `docker/Dockerfile.ref`.** The container stays as the documented,
  reproducible reference environment. This plan adds a second way to run the
  same scripts, it does not replace the first.
- **Regenerating the whole parity table on ROCm and calling it canonical.**
  See D1 — that is a separate decision, and this plan deliberately stops short
  of it.
- **Anything about the 1536 tier's quality.** That is settled and recorded in
  `docs/progress/1536-cascade_backend-limits.md`.

## Architecture decisions

**D1 — A ROCm-generated reference is authoritative for structure, not for
numerics.** This is the load-bearing decision and the reason the plan is
staged the way it is.

The parity method assumes the reference is *independent* of the port. If both
run on ROCm, a shared backend defect cancels: the test goes green while both
sides are wrong. That is not hypothetical here — the `mul_mat` column ceiling
(`docs/bugs/ggml-rocm-mul-mat-column-limit.md`) is exactly such a defect, and a
ROCm reference would carry it too.

So:

- **Integer/structural captures are unaffected.** `hr_coords_1536` and
  `hr_resolution_1536` come out of `((c + 0.5) / lr_res * grid).int()` — a
  quantization with no precision question, compared with a 5e-4 set tolerance
  for subdivision-boundary flips anyway. A ROCm run is a valid reference for
  these.
- **Numeric taps are not.** DINO at 7e-7 and SLAT at 6e-7 are tight enough that
  a shared-backend bias would hide in them.

Consequence: numeric rows regenerated on ROCm must say so in
`docs/VERIFICATION.md`, as a second measurement beside the documented CUDA
baseline rather than in place of it. A number that proves less than it appears
to is worse than no number.

**D2 — Skip the HR sampler for the coordinate captures.** `hr_coords_1536`
depends only on `up_coords`, which exists *before* the HR flow:

```text
ss_sample_ref.bin -> 512 flow -> lr_slat -> dec.upsample(x4) -> up_coords -> integer quantization
```

The HR sampler afterwards produces `hr_slat` and `hr_flow_t500_out`, neither of
which the 1536 rows use. A `--skip-hr-sampler` flag makes the cheap target
reachable without paying for the expensive one, on any backend.

**D3 — A partial dump must degrade, not break.** `test_cascade` already
tolerates a missing `hr_coords_1536` (it falls back to invariants and says so).
Sections 2 and 3 must gain the same behaviour, otherwise a dump written with
D2's flag turns a working test red. Reporting "not in the dump" is the correct
outcome for a tensor that was deliberately not produced; a `SKIP` is still not
a pass.

**D4 — TF32 handling on ROCm is unverified.** `scripts/ref_common.py` disables
TF32 because PyTorch's CUDA default costs ~10 bits of mantissa, and
`docs/VERIFICATION.md` records that this trap once made a correct port look
like a 0.08-rel-L2 bug. ROCm has no TF32, but whether those CUDA-specific
switches are no-ops or have an AMD-side equivalent has **not been checked**.
Phase 1 checks it before any numeric tap is trusted.

## Acceptance criteria

- [x] `scripts/dump_cascade_reference.py --skip-hr-sampler` produces a dump
      containing `up_coords`, `hr_coords`, `hr_coords_1536` and
      `hr_resolution_1536`, and omits `hr_slat` / `hr_flow_t500_out` cleanly.
      — 9 tensors, 41 MB.
- [x] `test_cascade` against that dump: 1024 scaffold at the existing tolerance,
      1536 scaffold and resolution matched to the reference, sections 2 and 3
      reported as "not in the dump" rather than failing. — PASS, exit 0.
- [x] `docs/plan/1536-cascade.md` criterion 2 moves from `[~]` to `[x]`, with
      the backend it was produced on named. — done, named as ROCm/R9700, with
      the caveat that this fixture stays under the token budget so the
      *reduction* branch remains unreferenced. Criterion 5 stays `[~]` on
      purpose: its 1024 coordinate rows were re-measured, its HR flow forward
      and 1024³ decode were not, because the dump is partial by design.
- [x] The dependency set needed to run the scripts natively is written down
      where someone can repeat it (a `docs/` note or a uv extra), not just in a
      shell history. — `pyproject.toml` (uv extras `rocm`/`cuda`/`cpu`),
      `.python-version`, `docs/reference-environment.md`, and
      `scripts/ref_env_check.py` as the gate that verifies a host.
- [x] `ctest --test-dir build -C Release -LE model` still green. A `SKIP` (77)
      is not a pass. — **10/10, no skips.** `preprocess` used to skip for want
      of the DINO dump's fixtures and now runs; `dual_grid` had never been built
      into this tree at all. The full suite (with `model`) is 13 passed,
      2 skipped, 3 failed — see below.

## Context / affected files

| Path | Why relevant |
|---|---|
| `scripts/dump_cascade_reference.py` | already captures the 1536 tensors; gains `--skip-hr-sampler` |
| `scripts/ref_common.py` | the TF32/determinism setup D4 questions |
| `scripts/refgen.sh` | the container orchestration; a native equivalent is needed |
| `docker/Dockerfile.ref` | the source of truth for the Python dependency set |
| `tests/test_cascade.cpp` | D3: tolerate a partial dump |
| `docs/VERIFICATION.md` | rows must name the backend a reference came from (D1) |
| `docs/plan/1536-cascade.md` | criteria 2 and 5 |
| `docs/ideas/TRELLIS-2_rocm/` | the reference implementation the scripts import (gitignored) |

## Plan (phases)

- [x] **Phase 0 — Environment.** Translate `docker/Dockerfile.ref`'s pip set
      into the reviewer's uv project (a `ref` extra alongside `rocm`, Python
      3.12). Fetch `models/` (~7 GB, `scripts/download_models.sh`). Confirm
      `docs/ideas/TRELLIS-2_rocm/` is importable. Answer D4: read what
      `ref_common.setup()` does and check whether it has any effect on ROCm.
      **Deliverable: `import torch; torch.cuda.is_available()` true on the
      R9700, and one reference script running end to end.**
      **Done** — see `docs/progress/rocm-native-reference_0-environment.md`.
      D4 answered by measurement: the switches do change state on ROCm
      (`cudnn.conv.fp32_precision` even defaults to `'tf32'`), but on gfx1201
      they change no numbers — matmul/SDPA/conv3d sit at 1.1e-06/5.0e-07/3.6e-07
      versus float64 before and after. Precision is settled for this card; D1 is
      not. `ctest -LE model` went from 8 passed + 1 skip to **9/9, no skips**.
- [x] **Phase 1 — The cheap chain.** `dump_dino_reference.py`,
      `tests/ref_ss_flow.py` (CPU by design — true fp32 golden),
      `tests/ref_ss_sample.py`, `tests/ref_ss_dec.py`. These produce
      `fixture*.dinodata` and `ss_sample_ref.bin`, which
      `dump_cascade_reference.py` needs as input. Note per D1 that the taps
      these enable are ROCm-derived.
      **Done** — all three ran on the R9700; six f32 GGUFs converted alongside.
      `test_dino` fails against the new reference at 4.6e-04 (the reference was
      verified correct against float64); see the progress entry.
- [x] **Phase 2 — `--skip-hr-sampler` and the 1536 capture.** Implement D2 and
      D3, run the cascade dump, point `test_cascade` at it. This is the phase
      that closes criterion 2, and the only one whose result is unaffected by
      D1.
      **Done** — `test_cascade` PASS: 1536 scaffold exact at 27,540/27,540,
      sections 2/3 reported as not in the dump. See
      `docs/progress/rocm-native-reference_1-2-cheap-chain-and-1536.md`.
- [~] **Phase 3 — f32 GGUFs and the rest of the suite (optional).** Convert
      `slat_flow_f32`, `slat_flow_1024_f32`, `shape_dec_f32` (`--ftype 0`), then
      the full cascade dump and `dump_slat_reference.py` /
      `dump_texture_reference.py`. Every row this turns green is a ROCm
      measurement and must be labelled as one. Decide *before* running whether
      that is wanted — see D1.
      **Half done.** All six f32 GGUFs exist (`dino`, `ss_dec`, `ss_flow`,
      `shape_dec`, `slat_flow`, `slat_flow_1024`) — they were needed in phase 1
      to compare against anything at all, and a lossless conversion of a
      checkpoint is not a measurement, so D1 does not apply to them.
      Outstanding: `dump_slat_reference.py` and `dump_texture_reference.py`,
      which are what `slat` and `texture` skip for. Those two *are* D1-affected.

## What running the suite revealed

The plan's stated goal was to stop the suite being dark. It worked, and the
light showed five failures — none introduced here, all simply unobservable
before. They are recorded in
`docs/progress/rocm-native-reference_1-2-cheap-chain-and-1536.md` with the
CPU/HIP split per test; this is the tracking list.

**Fixed while the plan ran** (each measured before and after):

- [x] **ggml's HIP graph capture crashed the sampler** (`0xC00000FD`,
      intermittent — 2 of 5 runs of `examples/ss_sample.exe`). The workaround
      existed in `server/main.go` alone, which is why the demo pipeline survived
      while every test and example died. Moved into `init_best_backend()` so all
      consumers get it; `TRELLIS2_CUDA_GRAPHS=1` re-enables for investigation.
      The underlying stack usage is **not** diagnosed, only avoided.
- [x] **ggml's ROCm flash-attention kernel cost ~5 % per flow forward.**
      `sdpa_auto()` requested `GGML_PREC_F32` all along; nothing under
      `ggml-cuda` reads it for this op. Attention is size-gated again: exact
      below a 1 GiB score matrix, flash above, with a free-VRAM check that can
      only lower exact usage. `ss_flow_forward` 5.198e-02 → 3.050e-06 and
      `ss_sample` 2.847e-01 → 2.135e-03, both green.

**Still open — no plan of their own yet:**

- [ ] **`chunked_decode` fails for every non-default block size**, including a
      voxel-count mismatch (858 vs 855). The default block size is bit-identical,
      which is the lever for bisecting it. This fork's own feature and the
      closest thing to a regression against `e87069e` (encoder levels chunked).
      Highest-value next step.
- [ ] **`dino`'s patch embedding is ~f16 accurate** (`embd` 4.618e-04 → `cond`
      1.623e-04) against a reference verified correct to float64. Identical on
      CPU and GPU, and unchanged by graphs or attention path, so it is the port
      rather than a backend. `ggml_conv_2d`'s F16 im2col is the suspect and is
      **not** verified. Note `ss_dec`'s dense 3D convolutions measure 6.172e-07
      on the same build, so it is not a blanket "ggml convolutions are f16".
- [ ] **`test_ss_dec` loads the decoder on the GPU** where `trellis2_capi.cpp`
      pins it to the CPU, so it hits the `CONV_3D` gap the shipped code never
      reaches (`docs/PLAN.md` documents that stage as CPU-only by design). One
      argument fixes the test — but it also hides that the stage has no GPU
      path, which is a reviewer's call, not an agent's.
- [ ] **Upstream ggml:** `ggml_flash_attn_ext_set_prec()` is silently ignored by
      the CUDA/HIP backend, and there is no F32 flash kernel to route to. The HR
      cascade therefore still carries the ~5e-02, since exact is not an option at
      its token counts. Worth reporting upstream independently of this fork.

## Tests / verification

- `test_cascade` with the partial dump: 1024 scaffold within 5e-4, 1536
  scaffold and resolution exact, sections 2/3 skipped with a reason.
  — **PASS**: upsample sym-diff 0.0002 %, 64³ scaffold 0.0000 %, 1536 scaffold
  27,540 of 27,540 at resolution 1536, sections 2/3 reported as absent.
- `ctest -LE model` green throughout (it is 10/10 today). — **10/10.**
- A partial dump and a full dump must both work; the flag changes what is
  produced, not whether the test runs. — the partial path is verified; a full
  dump has **not** been re-run since `--skip-hr-sampler` was added, so that half
  of the contract rests on the code shape rather than a measurement.

## Risks / open questions

- ~~**D4 is unanswered.**~~ **Answered in phase 0, favourably.** The switches
  are not silent no-ops (they flip `cudnn.allow_tf32`, the `fp32_precision`
  strings, and the SDPA backend selection), and measured against float64 the
  R9700 is at true fp32 both before and after them. Note `cudnn.conv.fp32_precision`
  defaults to `'tf32'` on ROCm and still measures 3.6e-07 — RDNA4 has no
  reduced-precision fp32 path, so the setting is advisory *on this card*. On
  CDNA (xf32) it would not be; rerun `scripts/ref_env_check.py` there.
- **Windows ROCm wheels are young.** The reviewer's own template records that
  AMD's Windows torch lacks distributed support. Other gaps may surface; the
  reference scripts are inference-only, which is the favourable case.
- **`docs/ideas/` and `models/` are gitignored**, so this environment cannot be
  reconstructed from the repository alone. That is exactly why Phase 0's
  deliverable includes writing the dependency set down.
- ~~**7 GB download** before anything runs.~~ Done: **16.3 GB** in practice
  (15 GB `TRELLIS.2-4B` alone) — the 7 GB estimate predates the texturing stack
  in `scripts/download_models.sh`.
- **Shared-backend cancellation is the real cost of this plan**, and it does not
  go away by being careful. The honest outcome may be that only Phase 2 is
  worth doing, and the numeric taps keep waiting for a CUDA machine.

## State this plan inherits

The state listed here when the plan was written — encoder chunking, the
subdivision-runaway warning, the dual-grid invariant test, and the server's
sampling-parameter round-trip — has since been committed (`e87069e` and
earlier). See `docs/progress/1536-cascade_backend-limits.md` for that account.

Other 1536 items still open and **not** covered by this plan:

- the reduction path has never fired in a real run (`0 reduction(s)` every
  time); forceable with `T2_MAX_NUM_TOKENS=20000`
- `decode_vram_peak`'s 1536 entry is still D4's extrapolated 15 GB — bounded
  by "whole run under 16 GB", never instrumented
- host RAM on the export side unmeasured (1536-cascade plan D5)
- where between 3.5M and 9.7M output voxels the subdivision starts to run away

## Release note

None user-facing. Developer tooling: the PyTorch reference dumps can be
produced on an AMD card through the native ROCm PyTorch build, instead of only
in the CUDA container.

## Proposed commit message

```text
Plan producing the reference dumps natively on ROCm

The parity suite is dark on the development machine - no models, no dumps - and
the CUDA reference container cannot use its Radeon card. AMD's Windows ROCm
wheels can, and ROCm PyTorch answers to device "cuda", so the scripts need no
change to run there.

The plan stages around one decision: a reference produced on the same backend
as the port is authoritative for integer captures and not for numeric taps,
because a shared backend defect cancels instead of showing up. That is enough
to close the 1536 coordinate criterion cheaply and not enough to regenerate the
parity table.

Plan only - no code.
```
