# Progress — `rocm-native-reference`, phase 0: Environment

- **Issue:** none — tracked by plan key
- **Plan:** [`docs/plan/rocm-native-reference.md`](../plan/rocm-native-reference.md) — Phase 0
- **Branch:** `resolution-1536`
- **Date:** 2026-08-14
- **Commits:** <to be added after review>

## Goal of the phase

Make the PyTorch reference scripts runnable on this machine's Radeon AI PRO
R9700 without Docker, write the dependency set down so it can be repeated, and
answer decision D4 — whether `ref_common.setup()`'s precision switches mean
anything on ROCm.

Deliverable per plan: `torch.cuda.is_available()` true on the R9700, and one
reference script running end to end.

## What was done

### A uv project at the repository root

`pyproject.toml` mirrors `docker/Dockerfile.ref`'s pip set as the base
dependencies and adds the torch build per backend extra (`rocm`, `cuda`, `cpu`,
mutually exclusive via `tool.uv.conflicts`). `package = false` — there is
nothing to build here, only dependencies. `.python-version` pins 3.12, which is
what AMD publishes the Windows ROCm wheels for.

Windows ROCm pulls torch/torchvision plus the runtime packages (`rocm`,
`rocm_sdk_core`, `rocm_sdk_libraries_custom`) from
`repo.radeon.com/rocm/windows/rocm-rel-7.2`. `.venv/` and `uv.lock` are already
gitignored, so the versioned artifact is the dependency set, not the resolution.

Installed: torch 2.9.1+rocmsdk20260116, HIP 7.2.26024, transformers 4.57.1.

### `scripts/ref_env_check.py`

One command that says whether this host can produce trustworthy dumps. It
reports the backend, checks `trellis2` imports through `TRELLIS2_PY`, lists what
`_force_true_fp32()` changes, and **measures** matmul / SDPA / conv3d against a
float64 ground truth — in the backend's default state *and* after the switches
are applied. Reading the flags was not enough to answer D4: a flag can be
settable and ignored.

### `docs/reference-environment.md`

The setup, the `TRELLIS2_PY` requirement, the native equivalent of
`scripts/refgen.sh`'s command sequence, the measured precision table, and the D1
warning. `docs/VERIFICATION.md`'s one-time-setup section now points at it.

## D4 — answered

**The switches are not no-ops as flags:**

| switch | ROCm default | after `_force_true_fp32()` |
|---|---|---|
| `cuda.matmul.allow_tf32` | `False` | `False` |
| `cudnn.allow_tf32` | `True` | `False` |
| `cuda.matmul.fp32_precision` | `'none'` | `'ieee'` |
| `cudnn.conv.fp32_precision` | **`'tf32'`** | `'none'` |
| `cuda.flash_sdp_enabled()` | `True` | `False` |
| `cuda.mem_efficient_sdp_enabled()` | `True` | `False` |
| `cuda.math_sdp_enabled()` | `True` | `True` |

**But on gfx1201 they change no numbers.** rel-L2 versus float64:

| op | cpu | cuda, default | cuda, after setup |
|---|---|---|---|
| matmul | 2.3e-07 | 1.15e-06 | 1.15e-06 |
| sdpa | 3.7e-07 | 5.10e-07 | 5.01e-07 |
| conv3d | 1.9e-07 | 3.62e-07 | 3.62e-07 |

The interesting one is conv3d: `cudnn.conv.fp32_precision` reports `'tf32'` by
default, and the result is still 3.6e-07. RDNA4 has no TF32-class path for fp32,
so the setting is advisory and the hardware runs full width either way.

Conclusion: the *precision* half of D4 is closed — numeric taps produced on this
card are at reference precision. What still stands against them is D1
(shared-backend cancellation), which is a property of the method and cannot be
measured away. **This result is specific to gfx1201.** CDNA (MI300, gfx942) does
have an xf32 path where that `'tf32'` conv default would really cost mantissa;
rerun `ref_env_check.py` on any other card.

## Affected files

- `pyproject.toml` — new; uv project, Dockerfile.ref's pip set + backend extras
- `.python-version` — new; pins 3.12
- `scripts/ref_env_check.py` — new; backend/import/precision gate
- `docs/reference-environment.md` — new; the native workflow and the D4 result
- `docs/VERIFICATION.md` — one-time setup points at the native path

No C++, no changes to `scripts/ref_common.py`, `scripts/refgen.sh` or
`docker/Dockerfile.ref`.

## Deviations / fixed along the way

- **`triton-rocm` had to be declared explicitly.** The first `uv sync` failed —
  not on Windows, but on the Linux split of the `rocm` extra: the Linux ROCm
  torch wheels hard-depend on `triton-rocm`, which exists only on the PyTorch
  ROCm index. Declared in the extra with a matching `tool.uv.sources` entry.
- **`ref_common.py` was left alone.** Its `TRELLIS2_PY` default of `/trellis2`
  is correct for the container; natively the variable is set explicitly and
  documented. A native `refgen` wrapper that sets it belongs to a later phase.
- **Phase 1's import risk was retired early.** `ref_common.setup()` plus the
  full model-import chain (`SparseStructureFlowModel`, `SLatFlowModel`,
  `FlexiDualGridVaeDecoder`, `samplers`, `SparseStructureDecoder`) imports
  cleanly against the ROCm checkout with `Conv backend: none, Attention backend:
  sdpa`. No CUDA/HIP extension is needed, as designed.

## Verification

**Backend.** `torch.cuda.is_available()` is true on the R9700 (gfx1201,
31.9 GiB), torch 2.9.1+rocmsdk20260116, HIP 7.2.26024. The plan's Phase 0
deliverable, first half.

**One reference script end to end.** `scripts/dump_dino_reference.py` against
the standard fixture (`assets/example_image/0a34fae7…webp`), on `--device cuda`:

```text
cond:      shape=(1, 1029, 1024) mean=-0.000000 min=-29.3356 max=24.9390 l2=1026.4866
cond_1024: shape=(1, 4101, 1024) l2=2049.0906
wrote dumps/reference_dino.gguf (166,979,296 bytes), 42 tensors
```

Both shapes match what `AGENTS.md` documents for the conditioning tensor
(`[1, 1029, 1024]`) and the 1024-stage conditioning (4101 tokens). 42 taps, the
count the parity table expects for `test_dino` (40 taps plus inputs). Second
half of the deliverable.

**Model weights.** `scripts/download_models.sh` completed: 15 GB
`TRELLIS.2-4B`, 1.2 GB `dinov3-vitl16`, 141 MB `TRELLIS-image-large` — 16.3 GB,
noticeably more than the ~7 GB the plan estimated, because the plan's figure
predates the texturing stack in the download list.

**Test suite.** `ctest --test-dir build -C Release -LE model`: **9/9 passed, no
skips.** Before this phase the same command reported 8 passed + `preprocess`
skipped, because `test_preprocess` compares against `dumps/fixture_rgba.png` and
`dumps/fixture_512.png` — exactly the two files the DINO dump writes. That skip
is now a real pass, which is the first thing on this machine the parity method
actually verifies.

**Not verified.** `test_dino` still skips: it wants `ggufs/dino_f32.gguf` and
only `dino_f16.gguf` exists. Converting it is Phase 3, and per D1 the tap it
would produce is a ROCm measurement that has to be labelled as one.

## Open points

- **`test_dino` and the rest of the model-labelled suite stay dark** until the
  f32 GGUFs exist (Phase 3). Producing the reference is now possible; comparing
  against it still needs the f32 conversion.
- **The build tree is stale.** `ctest -LE model` runs 9 tests; the plan records
  10. `tests/test_dual_grid.cpp` and the matching `tests/CMakeLists.txt` change
  are uncommitted and not configured into `build/`. Not rebuilt here — the
  build scripts wipe `build/`, and this phase touched no C++.
- **ROCm has to be on `PATH` for ctest.** Without
  `C:\Program Files\AMD\ROCm\6.4\bin` the shared-library tests fail with
  `0xc0000135` (DLL not found), which looks like six test failures and is not
  one. Pre-existing and unrelated to this phase, but worth knowing.
- Everything D1 warns about is untouched: this phase makes numeric taps
  *producible*, not *authoritative*.

## Next phase

Phase 1 — the cheap chain (`dump_dino_reference.py` is already done here;
`tests/ref_ss_flow.py` on CPU, `ref_ss_sample.py`, `ref_ss_dec.py`), producing
`fixture*.dinodata` and `ss_sample_ref.bin` as input for the cascade dump. The
import chain those need is already verified.

## Proposed commit message

```text
Run the PyTorch reference scripts natively, without the CUDA container

The reference container is CUDA-only, so the Radeon development machine could
not produce dumps at all. A uv project at the root carries Dockerfile.ref's pip
set with the torch build behind a backend extra, which gets torch 2.9.1+rocm
onto the R9700 through AMD's Windows wheels. The dump scripts need no change:
ROCm PyTorch answers to device "cuda".

scripts/ref_env_check.py is the gate before trusting anything produced this way.
It measures matmul, SDPA and conv3d against a float64 ground truth rather than
reading the precision flags, which answers D4: ref_common's switches do change
state on ROCm - cudnn.conv.fp32_precision even defaults to 'tf32' - but on
gfx1201 they change no numbers, because RDNA4 has no reduced-precision fp32
path. That closes the precision question and not the shared-backend one.
```
