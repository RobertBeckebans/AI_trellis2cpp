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

**D1 addendum, 2026-08-17 — it happened, in the mirror image, and that is the
worse direction.** The failure mode anticipated above is a *shared* defect that
cancels and turns a test green while both sides are wrong. What phase 3 actually
hit is a defect only the **reference** carries, which turns a test red and reads
as a port defect. `tests/test_slat` failed `out7` at rel-L2 1.103 and
`lvl4.pre_up` at 0.979 against a ROCm-generated reference; the same taps come
out at 4.3e-07 and 4.0e-07 against a CPU-generated one, 18 of 18 green. The port
was correct throughout.

Two things make this direction harder than the one D1 was written for. A green
test invites no investigation, so a shared defect costs nothing until someone
looks; a red one sends the search into the port, where it finds nothing, for as
long as it takes to doubt the reference. And the yardstick here is not upstream
at all: `scripts/ref_common.py` monkey-patches its own sparse convolution in as
backend `"none"` and hard-assigns `sp.config.CONV`, so at that stage the parity
test compares two of this project's implementations and calls one of them the
reference.

So the consequence above is not enough, and is superseded: **numeric reference
dumps are produced on the CPU on this hardware.** Not labelled as ROCm-derived —
not produced on ROCm. The cost is ~75 min against ~4 for the SLAT dump, and most
of it is the sampler, which the decoder taps do not need. See
`docs/progress/rocm-native-reference_3-slat-dump-and-out7.md`.

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
      **Mostly done.** All six f32 GGUFs exist (`dino`, `ss_dec`, `ss_flow`,
      `shape_dec`, `slat_flow`, `slat_flow_1024`) — they were needed in phase 1
      to compare against anything at all, and a lossless conversion of a
      checkpoint is not a measurement, so D1 does not apply to them.

      `dump_slat_reference.py` is **done**, and is what produced the D1 addendum
      above: `test_slat` is **18 of 18 green** against a CPU-generated reference
      (`--device cpu`), where the ROCm-generated one failed `out7` at 1.103. It
      also gained the taps that made the bisection possible — the finest level's
      `pre_up`, the up-block's `in_hch` / `in_xch`, and its `norm2` / `conv2`.
      The last three are investigation aids and should come back out; only
      `pre_up` earns a permanent place.

      Outstanding: `dump_texture_reference.py` (input pair identified, but
      `tex_slat_flow_512` exists only as f16), and the **full** cascade dump
      without `--skip-hr-sampler`, which `backend-parity` phase 1b needs. Both
      on the CPU.

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

- [~] **`chunked_decode` fails for every non-default block size** on HIP,
      including a voxel-count mismatch (858 vs 855). **Diagnosed: the chunking
      is correct and the backend is not shape-stable.** On CPU, with the same
      f16 weights, chunked equals unchunked *exactly* — `max|d| = 0` in all
      twelve cases — so the claim the test header makes (norm and silu are
      per-row and commute with the neighbour gather) holds, and `res` really is
      the full `[Cin, L]` input every block gathers from. The divergence is
      ggml's ROCm kernels returning different results for different matrix
      shapes, and it scales with weight precision:

      | | CPU f16 | GPU f32 | GPU f16 |
      |---|---|---|---|
      | texture decoder | 0 | ≤ 2.03e-06 | 1.19e-03 |
      | shape decoder | 0 | 7.25e-05 (`block=37` only) | voxel count changes |
      | shape encoder | 0 | ≤ 5.72e-06 | 3.86e-03 |

      Not a regression against `e87069e` — that suspicion is withdrawn. What
      remains is a decision rather than a fix: the test asserts bit-identity,
      which is right on CPU and unattainable on HIP. Either the gate becomes
      backend-aware, or the shape-dependence is chased into ggml.

      **2026-08-16, two corrections to the paragraph this replaces.** It said
      "at f16 on ROCm the chunk size changes the mesh, and the chunk size comes
      from the backend's column limit, so production output depends on it."
      Neither half holds. Measured at 1024: taking the encoder's level 1 out of
      the chunked path leaves the output **bit-identical**, PBR block included,
      so the 3.86e-03 above is a property of `block=37` on a small model rather
      than of production shapes. And the block size does not come from the
      column limit: `min(chunk_rows_limit(), max_rows)` is `min(2^18, 2^19|2^21)`,
      so the limit never binds and the block is always 262,144 — which is
      `512 x 512`, ggml-vulkan's 2D dispatch plane, and half of the 524,288 that
      upstream's own `ROCM_SAFE_CHUNK` calls confirmed-safe on this GPU.
- [~] **Does f16 weight noise fray the geometry?** The question that follows
      from the entry above: the `to_subdiv` head is thresholded at zero, and f16
      leaves ~1e-3 of noise on those logits where f32 leaves ~1e-6. Measured
      with `tools/gguf_precision_ab.sh` — one image, seed 1234, 1024³, three
      weight configurations, reading the `[mesh]` line the C-ABI prints under
      `TRELLIS2_TIMING`:

      | | verts | boundary edges | non-manifold | L4/L3 |
      |---|---|---|---|---|
      | all f16 | 2,572,004 | 195,049 (2.6321 %) | 62,095 | 3.894 |
      | shape_dec f32 | 2,571,970 | 194,688 (2.6273 %) | 62,018 | 3.894 |
      | all f32 | 2,968,528 | 131,909 (1.5586 %) | 54,812 | 3.949 |

      **The decoder is not the cause.** The middle row is the controlled
      comparison — identical latent, verified by voxel-identical decode levels,
      only the decoder's weights swapped — and it moves the boundary fraction by
      0.0048 points. The one spare gigabyte buys nothing.

      **The bottom row is not evidence.** It looks dramatic (41 % fewer boundary
      edges) but f32 weights in DINO and the flows send the sampler down a
      different trajectory despite the same seed, so it is a *different object*:
      2.97M vertices against 2.57M. At n = 1 that could equally be a denser
      sample meshing more cleanly. **Do not cite this as "f32 makes better
      meshes."** Settling it needs two or three more seeds per configuration, or
      a sharper cut — flows in f32, decoder in f16 — to ask whether precision
      matters upstream of the decoder at all.
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
- [ ] **Vulkan is not usable above 1024, and degrades below it.** Measured on the
      same card, same image, same seed, with the runtime backend switch:

      | | ROCm | Vulkan |
      |---|---|---|
      | 1024, boundary edges | 3,534 (0.042 %) | 15,171 (**0.172 %**) |
      | 1024, non-manifold | 22,814 | 53,593 |
      | 1024, expansion | 3.997 | 4.008 |
      | 1024, total time | 321 s | **232 s** |
      | **1536** | (not run) | **collapsed** |

      At 1536 it falls apart rather than degrading: finest-level expansion
      **2.820** where a surface gives ~4, and 163,318 triangles for 2,183,769
      vertices — a fifteenth of the usual ratio — with 16.06 % boundary edges.
      The viewer shows loose fragments, not a mesh. The corruption starts well
      before the finest level: levels grow 6.3× → 2.5× → 1.7× → 2.8× instead of
      roughly 4× each.

      The chunking was active (`split block 262144`), so this is *not* the
      mul_mat column ceiling. Vulkan has a second limit of its own — it asserted
      on `maxComputeWorkGroupCount` under `TRELLIS2_NO_CHUNK`, so evidently not
      every op checks it and some produce garbage silently instead.

      **The two backends fail differently, and only one of them coherently.**
      Same image and seed at 1536:

      | | ROCm | Vulkan |
      |---|---|---|
      | level growth | 4.08 → 4.21 → 4.24 → **4.51** | 6.33 → 2.49 → **1.73** → 2.82 |
      | verts / tris | 9,247,597 / 22,122,300 | 2,183,769 / **163,318** |
      | tris per vert | 2.39 | **0.075** |
      | boundary edges | 0.69 % | **16.06 %** |
      | non-manifold | **9.87 %** | 1.29 % (meaningless at that tri count) |

      ROCm subdivides consistently and only overshoots at the finest level —
      the known runaway from `1536-cascade_backend-limits.md`, worse than the
      7.47 % recorded there, and the built-in early warning **fired correctly**.
      Vulkan's growth rates are incoherent from level 2 onward: it collapses
      long before the finest level, and the extractor then finds almost no
      closed surface.

- [ ] **The subdivision-runaway warning is one-sided and missed the worse case.**
      It fires on `ratio > 4`. Vulkan's 2.82 sits *below* four, so nothing was
      logged for the run that failed far harder. A ratio well under 4 is equally
      diagnostic — the decoder is losing voxels rather than adding volume — and
      `tris/verts` (2.39 healthy, 0.075 collapsed) is an even cheaper tell that
      nothing currently reads. Making the check two-sided would have put both
      failures in the log instead of leaving one to the eye, which is what the
      warning exists for.

      At 1024 the trade is real and worth knowing: Vulkan is 28 % faster (the
      flow samplers run 3-4× faster) and produces four times the boundary edges.
      Neither backend is simply better.
- [ ] **Upstream AMD: the mul_mat column truncation is rocBLAS, not ggml.**
      Reproduced with PyTorch alone — `tools/rocblas_col_truncation.py` —
      `torch.matmul` leaves the tail of a tall narrow GEMM as exact zeros with
      no error: f32 past 2^19, f16/bf16 past 2^22, clean at `K = M = 256`, clean
      in f64. This is what the decoder chunking works around, and it is the one
      finding here that reaches past this project.

      Two things fall out of it. **ROCm 7.2 does not fix it** — that measurement
      ran on `torch.version.hip` 7.2.26024, newer than the 6.4 this project
      builds against, and the f32 cap is the same 2^19, so an SDK upgrade is not
      the way out and the chunking stays. And the existing note
      `docs/bugs/ggml-rocm-mul-mat-column-limit.md` blames ggml for something
      ggml does not own; it now carries the PyTorch measurement and says so.

      **Vulkan on the same GPU does not truncate.** `tests/test_large_rows`
      built from the same ggml commit against Vulkan instead of HIP reports no
      break for f32, f16 or bf16 where HIP breaks at 2^19 / 2^21 / 2^21. So it
      is not the hardware either, and the chain is closed: CPU correct → not the
      port; PyTorch on ROCm equally broken → not ggml; Vulkan correct → not the
      silicon. Note the failure modes differ in kind — Vulkan's f16 is merely
      imprecise (~2.6e-03 across the whole matrix), HIP's is bit-identical and
      then exactly zero. Only the second is silent.

      Unverified but worth checking first: 2^19 = 65536 × 8 and 2^22 = 65536 × 64
      exactly, which suggests a grid dimension pinned at 65535 times the tile
      height rather than an arithmetic overflow.

      The Vulkan tree lives in `build-vulkan/` (configured by hand, not via
      `cmake-ninja-win64-vulkan.bat`, which wipes `build/`). Worth keeping until
      the ticket is filed.

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

- **`ref_common`'s own sparse convolution is wrong on ROCm, and it is not
  pinned to a line.** Same code, same weights, same model: on the CPU it matches
  a direct reimplementation at cos +1.000000, on ROCm at +0.164 with 4.9x the
  amplitude, while its input (`norm2`) and its composition
  (`pre_up = conv2 + skip`) check out at +1.000000 on both. Leading suspect is
  the masked scatter-add `out[hit] += contrib`, which goes through atomics on
  the GPU and would explain the amplitude *and* the run-to-run drift
  (`intersected frac` 0.5654 → 0.5628 → 0.5527 across three runs at one seed).
  `torch.searchsorted` is the second candidate. A minimal reproducer would be
  reportable upstream and would say whether anything in the port touches the
  same op.
- **`ref_common.setup()` overrides `SPARSE_CONV_BACKEND` rather than honouring
  it.** Line 35 sets it as a default, line 260 hard-assigns `sp.config.CONV`
  after registering its own module as backend `"none"`. So the documented knob
  does nothing, `trellis2/modules/sparse/conv/conv_none.py` never runs, and an
  attempt to compare against a real upstream backend silently measures the same
  path twice. Worth fixing before any backend comparison is attempted here.
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
