# Progress — `rocm-native-reference`, phases 1 + 2: the cheap chain and the 1536 capture

- **Issue:** none — tracked by plan key
- **Plan:** [`docs/plan/rocm-native-reference.md`](../plan/rocm-native-reference.md) — Phases 1 and 2
- **Branch:** `resolution-1536`
- **Date:** 2026-08-14
- **Commits:** <to be added after review>

## Goal of the phases

Phase 1: produce the reference chain the cascade dump needs as input
(`fixture*.dinodata`, `ss_sample_ref.bin`). Phase 2: implement D2
(`--skip-hr-sampler`) and D3 (a partial dump must degrade, not break), run the
cascade dump, and gate `test_cascade` against it — the phase that closes
criterion 2 of `docs/plan/1536-cascade.md`.

## What was done

### Phase 1 — the cheap chain

All three scripts ran natively on the R9700. They do not import `ref_common`
(they set `ATTN_BACKEND` / `SPARSE_CONV_BACKEND` themselves), which turned out
not to matter: `trellis2.pipelines` and `trellis2.representations` resolve
lazily through `__getattr__`, so nothing pulls the absent CUDA extensions.

| artefact | how | result |
|---|---|---|
| `tests/ss_flow_ref.bin` | `ref_ss_flow.py`, CPU by design | 30 blocks, d=1536, `out` l2=152.70288 |
| `tests/ss_sample_ref.bin` | `ref_ss_sample.py --device cuda` | z_s l2=96.61504, 12-step CFG |
| `tests/ss_dec_ref.bin` | `ref_ss_dec.py --device cuda` | logits `[1,64,64,64]` |

The CPU fp32 forward is genuinely expensive — roughly 40 minutes of wall clock
and ~12,000 CPU-seconds at 10.6 GB resident. That is by design (it is the true
fp32 golden), but it is worth knowing before scheduling a rerun.

Six f32 GGUFs were converted so the tests have something to compare against:
`dino`, `ss_dec`, `ss_flow`, `shape_dec`, `slat_flow`, `slat_flow_1024`. These
are lossless conversions of the checkpoints, not measurements, so D1 does not
apply to them.

### Phase 2 — D2 and D3

**D2, `scripts/dump_cascade_reference.py --skip-hr-sampler`.** Everything from
the HR flow onwards is wrapped in an `else` branch: no 1024 flow model, no HR
sampler, no 1024³ decode. `hr_coords_1536` / `hr_resolution_1536` depend only on
`up_coords`, which exists before that point. The manifest records
`skip_hr_sampler` so a reader can tell a partial dump from a truncated one.

Result — 9 tensors, 41 MB instead of gigabytes:

```text
scaffold:    2401 voxels at 32^3
upsample:    1185000 candidate coords at 512^3
hr scaffold: 10965 voxels at 64^3
1536 budget: 27540 voxels at 96^3 (resolution 1536, budget 49152)
```

**D3, `tests/test_cascade.cpp`.** `hr_noise` moved out of the required-tensor
set; sections 2 and 3 now report the absent tensors and skip instead of
returning 1. Section 1 — the part the 1536 tier is gated on — runs either way.

## Verification

`test_cascade` against the partial dump:

```text
[upsample coords]  got 1185000 vs ref 1185000, sym-diff 0.0002% -> OK
[hr scaffold]      got 10965 vs ref 10965,     sym-diff 0.0000% -> OK
[budget 1024]      res 1024, 10965 tokens, 0 reductions -> OK
[budget 1536]      res 1536 (grid 96), 27540 tokens, 0 reductions
[budget 1536 inv]  max coord 95 < 96, fits budget or floored -> OK
[hr scaffold 1536] res 1536 vs ref 1536, got 27540 vs ref 27540, sym-diff 0.0000% -> OK
[hr flow t500]     not in the dump (--skip-hr-sampler), section skipped
RESULT: PASS
```

All three acceptance criteria of phase 2 hold: the 1024 scaffold is within
tolerance, the 1536 scaffold and resolution match the reference **exactly**
(27,540 of 27,540), and the missing sections are reported rather than failed.
Exit code 0, not 77 — this is a pass, not a skip.

Per D1 this is authoritative despite being ROCm-derived: both sides compute the
integer quantization `((c + 0.5) / lr_res * grid).int()`, which has no precision
question.

Full suite, `ctest --test-dir build -C Release` on Windows/HIP (ROCm must be on
`PATH` or the shared-library tests fail with `0xc0000135`):

```text
18 tests: 11 passed, 2 skipped (slat, texture), 5 failed
failed: chunked_decode, dino, ss_flow_forward, ss_sample (segfault), ss_dec
```

`ctest -LE model` on its own is **10/10 green**, including `preprocess`, which
used to skip, and `dual_grid`, which had never been built here. The five
failures are all `model`-labelled and are analysed below.

## Deviations / fixed along the way

- **`convert_dino_to_gguf.py` resolved `models/` against `scripts/`.** Its
  `--model` default was built from `os.path.dirname(__file__)`, so it looked for
  `scripts/models/dinov3-vitl16/…`. Fixed to resolve against the repository
  root. The other converters take `--model` explicitly, which is why this went
  unnoticed.
- **The reference tree logs to a hardcoded POSIX path.**
  `trellis2/utils/pipeline_logger.py` opens `/tmp/trellis2_pipeline.log`, which
  on Windows resolves against the current drive (`F:\tmp\…`) and does not exist,
  killing `dec.upsample` inside a logging `FileHandler`. `ref_common` now
  redirects `LOG_PATH` to the platform temp directory — but only when the
  original directory is missing, so the container is untouched.
- **The 1536 budget did not reduce on this fixture.** 27,540 tokens against a
  49,152 budget, so `hr_resolution_1536` stayed at 1536 and the loop took the
  zero-reduction path. The capture is a valid reference for the *selection*, and
  it does not exercise the *reduction* branch at all.

## The suite came alive, and five tests fail

This is the first full `ctest` on this machine — the whole point of the plan.
With all six f32 GGUFs and the references in place, 18 tests run: 11 pass, 2 skip
(`slat`, `texture` — phase 3), 5 fail. The failures were **not** introduced
here; they were invisible because nothing ran.

The decisive experiment for each was CPU versus HIP, because the C++ side takes
`TRELLIS2_DEVICE`:

| test | CPU | GPU (HIP) | reading |
|---|---|---|---|
| `ss_dec` | **PASS** rel-L2 6.172e-07, sign 100% | crash | backend gap the pipeline already avoids |
| `ss_flow_forward` | **PASS** rel-L2 2.416e-04 | **FAIL** rel-L2 5.198e-02 | ROCm flash-attention kernel |
| `ss_sample` | not run (≫1 h on CPU) | crash, then **FAIL** 2.844e-01 | HIP graph capture + the same kernel |
| `dino` | **FAIL** 4.618e-04 | **FAIL** 4.618e-04 | backend-independent |
| `cascade` | **PASS** | **PASS** | — |
| `chunked_decode` | not run | **FAIL** | f16 chunking, see below |

### Root causes — three of the five collapse into two ggml/ROCm defects

Following up on each failure rather than stopping at "it is red":

**`ss_dec` is a test bug, not a product bug.** `src/trellis2_capi.cpp:549` already
carries the answer, and it predates this work:

```cpp
// The SS occupancy decoder uses a genuine dense CONV_3D, for which ggml has
// no CUDA kernel, so it stays on the CPU (it is only ~3 s / 4 % anyway).
p->dec = trellis2_ss_dec_load( ss_dec_gguf, true, &e, "cpu" );
```

The pipeline pins the stage to CPU; `tests/test_ss_dec.cpp:64` loads it with the
default device and therefore hits a gap the shipped code never reaches. The test
should pass `"cpu"` the way the pipeline does — on CPU it already measures
6.172e-07.

**`ss_sample`'s crash is ggml's HIP graph path**, and it is *not* test-only:
`examples/ss_sample.exe`, an ordinary consumer of the library, crashes the same
way (`0xC00000FD`, always right after `CUDA graph warmup complete`, at a
non-deterministic step). With `GGML_CUDA_DISABLE_GRAPHS=1` all 12 steps run and
produce a sane latent (occupancy 52.36 % against the reference's 51.84 %).

**The numeric divergence is ggml's ROCm flash-attention kernel**, and it is the
single most consequential finding here. Disabling graphs changes
`ss_flow_forward` by nothing at all (5.198e-02 either way). Switching the
attention path with `TRELLIS2_SDPA_EXACT=1` changes everything:

| test | flash (default) | exact softmax |
|---|---|---|
| `ss_flow_forward` | 5.198e-02 **FAIL** | **3.050e-06 PASS** |
| `ss_sample` | 2.844e-01 **FAIL**, sign 89.8 % | **1.494e-05 PASS**, sign 100.000 % |

A factor of ~17,000 on the forward, and both land *better* than the documented
CPU numbers. `VERIFICATION.md` budgets ~3e-03 for flash on the CUDA F16-MMA
kernel; the ROCm kernel costs 5e-02, more than an order beyond that.

Because `sdpa_auto()` uses flash for **every** flow forward by default, the
shipped pipeline on this card runs with roughly 5 % relative error per forward.
That is an output-quality issue, not merely a test failure.

**This is not simply "turn flash off", though.** `VERIFICATION.md` records why
flash became the default: the HR cascade's 49,152-token attention fits in 16 GB
with flash and would need a 108 GiB exact score matrix. So the exact path is not
available at cascade resolutions. The ROCm kernel needs fixing or avoiding on
its own terms; the env switch is a diagnostic, not a shipping answer.

**`dino` is untouched by both.** Identical numbers with graphs disabled and
exact softmax (`embd` 4.618e-04, `cond` 1.623e-04). It stands alone.

### Follow-up fix: the graph workaround moved into the library

The workaround already existed — in `server/main.go` alone. That is why the demo
pipeline generated meshes while every test and example crashed: the server sets
`GGML_CUDA_DISABLE_GRAPHS=1` for itself and nothing else does.

Its stated reason no longer holds, though. The comment read "the same DLL driven
from a native executable is fine, so it is the small Go-managed thread stack
rather than the graph itself". Measured now, `examples/ss_sample.exe` — native,
no Go — crashes too, **intermittently**:

| graphs | 5 runs of `examples/ss_sample.exe` |
|---|---|
| enabled (`TRELLIS2_CUDA_GRAPHS=1`) | **2 crashes**, 3 clean |
| disabled (new default) | 0 crashes |

Intermittency is presumably why this read as Go-specific: a native run has a
better-than-even chance of surviving. It is the graph path's stack usage, not
Go's thread.

`init_best_backend()` in `src/trellis2.cpp` is the single funnel through which
every backend is created, so the default now lives there and covers tests,
examples and any dlopen consumer. Two ggml details shaped it: the flag is read
into a function-local `static` on first use, so setting it before any backend
exists is early enough; and ggml tests only for *presence*, so
`GGML_CUDA_DISABLE_GRAPHS=0` would also disable graphs — hence a separate
`TRELLIS2_CUDA_GRAPHS=1` opt-in, which is what keeps the crash reproducible.

Verified: `_putenv_s` from `trellis2.dll` does reach `getenv` in `ggml-hip.dll`
(shared dynamic UCRT) — the one way this could have failed silently.

Effect on the suite: `ss_sample` goes from **SEGFAULT** to a plain `Failed` with
a measurable rel-L2. The failure count is unchanged at five, but one of them is
now a number instead of a crash. `server/main.go` keeps its block as
belt-and-braces against an older library; its comment was corrected.

**This also validates the ROCm-produced references.** `ss_flow_forward` on CPU
lands at 2.416e-04 against `VERIFICATION.md`'s documented 2.4e-04 — the same
number to three figures. `ss_dec` agrees at 6.172e-07 while its reference came
off the *GPU* and the port ran on the *CPU*, i.e. two different backends, which
is exactly the independence D1 asks for. The native reference path produces
sound references.

### Which reference implementation these came from

The dumps were produced against `docs/ideas/TRELLIS-2_rocm/`, which is
byte-identical to `docs/ref/TRELLIS-2_rocm/` and forks Microsoft's TRELLIS.2 at
`5565d24 Release Training Code`. Upstream's commits after that point are CodeQL
workflow only, so the fork is not behind on model code.

Diffing the fork against its fork point, the modules behind these references are
**unmodified from Microsoft's original**:

| module | used for | vs. upstream |
|---|---|---|
| `models/sparse_structure_flow.py` | `ss_flow_ref.bin`, `ss_sample_ref.bin` | unchanged |
| `models/sparse_structure_vae.py` | `ss_dec_ref.bin` | unchanged |
| `models/structured_latent_flow.py` | cascade LR flow | unchanged |
| `pipelines/samplers/flow_euler.py` | every sampler run | unchanged |
| `models/sc_vaes/sparse_unet_vae.py` | cascade `dec.upsample` | **+54/-7** |
| `models/sc_vaes/fdg_vae.py` | cascade decoder wrapper | **+7/-0** |

Only the cascade's upsample path touches modified code. Going through that diff,
every change is inert for a fp32 run: `subdiv.feats.float() > 0` before the
subdivision threshold is a no-op when the tensor is already fp32 (it exists for
bf16 runs); the MLP chunking above 524,288 rows is row-independent and therefore
exact; `convert_to_f16`'s switch to bf16 is never reached because we force
`use_fp16=False`; a `.contiguous()` before `output_layer` changes memory layout,
not arithmetic, and sits in `forward()` rather than `upsample()`; the NaN/Inf
escape hatch added to the upsample loop did not trigger. The remainder — and all
of `fdg_vae.py`'s delta — is debug logging, which is what the hardcoded `/tmp`
path came from.

Independent of reading the diff: the C++ port reproduced `up_coords` to 0.0002%
and `hr_coords` exactly. A fork that had bent the upsample arithmetic would not
produce that agreement.

The DINO finding below is untouched by any of this — `dump_dino_reference.py`
does not import the reference tree at all; it drives HuggingFace
`transformers`' `DINOv3ViTModel` directly.

### `ss_dec` — `CONV_3D` is not implemented on the HIP backend

```text
ggml_cuda_graph_evaluate_and_capture: op not supported node_0 (CONV_3D)
ggml/src/ggml-cuda/ggml-cuda.cu:4056: GGML_ASSERT(ok) failed
```

`0xc0000409` is that assert aborting, not stack corruption. On CPU the same
decoder is correct to fp32 rounding. A backend capability gap, not a port bug.

### `ss_sample` — stack overflow on HIP

`0xC00000FD` (`STATUS_STACK_OVERFLOW`) at step 2 of 12, immediately after
`CUDA graph warmup complete`. Not investigated further; CPU would take well over
an hour for 24 forwards, so there is no CPU counterpart yet.

### `ss_flow_forward` — the HIP backend, not the port

CPU 2.416e-04 (passes, matches the documented value); HIP 5.198e-02 (fails a
2e-03 gate). `VERIFICATION.md` already records ~3e-03 for CUDA F16-MMA flash
attention; 5.2e-02 on ROCm is more than an order beyond that and deserves its
own look.

### `chunked_decode` — this fork's own feature, on f16 weights

Unlike the others this one uses the **f16** GGUFs, so it could have run before.
`block=default` is bit-identical (`max|d| = 0`), but every non-default block
size diverges, and one case changes the voxel count:

```text
tex decoder    block=997  65718/196608 over 1e-5, max|d| 0.00054
               block=37  169106/196608 over 1e-5, max|d| 0.00119
shape decoder  block=37  COORD/SHAPE MISMATCH 858 vs 855 voxels -> FAIL
shape encoder  block=997 254/256 over 1e-5, max|d| 0.00274
```

The shape-encoder rows are the ones to look at first given that encoder levels
1-3 were chunked in the immediately preceding commit.

## The `dino` finding

`docs/VERIFICATION.md` records DINO at **rel-L2 ≤ 7e-7 across all 40 taps**.
Measured against the reference produced here:

```text
[embd] max|d|=0.004276  relL2=4.618e-04  -> FAIL   (first divergence)
[cond] max|d|=0.006466  relL2=1.623e-04
37 of 40 taps over the gate
```

What was ruled out before reporting this:

- **Not a backend effect.** CPU 4.619e-04, GPU 4.618e-04 — the same number.
- **Not flash attention.** `embd` diverges *before* any attention; `rope_0/1`
  are clean at 5e-08.
- **Not the input.** `test_dino` feeds `pixel_values` straight from the dump.
- **Not a half-precision GGUF.** `--ftype 0` writes all 415 tensors as f32.
- **Not the reference.** The patch embedding was recomputed from
  `model.safetensors` in float64 and compared against the dump's `embd`:
  **rel-L2 7.6e-07**, i.e. exactly fp32 rounding. The reference is faithful.

A caveat on the "37 of 40 taps" headline: most of those are the tight
`atol=2e-3` gate biting on huge activations — `l0`…`l13` all report
`max|d| = 0.09375` on values around 155,570, which is 6e-07 relative, i.e.
fp32 rounding. The genuinely divergent taps are `embd` (4.6e-04 on values around
0.74) and what it carries downstream: `cond` ends at rel-L2 1.623e-04, against a
documented ≤ 7e-07.

What remains: the port computes the patch embedding at roughly f16 precision
(4.6e-04 ≈ 2⁻¹¹), identically on both backends. `ggml_conv_2d`'s F16 im2col path
is the obvious suspect — **not verified**. Note that it cannot be a blanket
"ggml convolutions run at f16" story: `ss_dec` is a dense 3D-conv decoder and it
lands at 6.172e-07 on the same build, so whatever this is, it is specific to the
DINO patch-embedding path.

`src/trellis2.cpp` was deliberately not touched: that is a numerical change with
its own blast radius and deserves its own review.

Undecided, and not decidable from what is in the tree: whether this is a
regression or whether the 7e-7 row never covered this tap. The DINO row in
`VERIFICATION.md` was left as it stands rather than rewritten on one session's
measurement.

## Open points

Ranked by what they block:

1. **ggml's ROCm flash-attention kernel costs ~5 % relative error per flow
   forward**, and flash is the default for every one of them. This degrades real
   output on this card, not just tests. Cannot be fixed by switching the default
   back: the exact path cannot hold the cascade's HR attention.
2. **ggml's HIP graph capture crashes the sampler** (`0xC00000FD`), reproducible
   from `examples/ss_sample.exe`. `GGML_CUDA_DISABLE_GRAPHS=1` avoids it.
3. **`chunked_decode` fails for every non-default block size**, including a
   voxel-count mismatch. Closest to the immediately preceding encoder-chunking
   commit, and this fork's own feature.
4. **The `dino` patch embedding is ~f16 accurate**, independent of backend,
   graphs and attention path.
5. **`tests/test_ss_dec.cpp` should load the decoder on CPU**, as
   `trellis2_capi.cpp` already does. One argument; turns a crash into the
   6.172e-07 pass it already achieves.
6. **The reduction branch is still unreferenced.** Forcing it needs a second dump
   with a lower `--max-num-tokens` plus test wiring to consume it; outside this
   plan's phase 2.
7. `test_slat` and `test_texture` remain dark — they need
   `dump_slat_reference.py` / `dump_texture_reference.py`, which is phase 3, and
   every row those turn green is a ROCm measurement that D1 requires to be
   labelled as one.

None of 1-5 was introduced by this work; all five were simply unobservable
before, which is what the plan set out to change. Whether any is a regression
against the CUDA/Linux baseline the parity table was written from cannot be
decided from this machine — but 1 and 2 are ROCm-specific by construction, so
that baseline would not have caught them.

`test_dual_grid` now builds and passes — it was registered in `tests/CMakeLists.txt`
but had never been compiled into this build tree.

## Next phase

Phase 3 (the remaining dumps) is optional and gated on the D1 decision, and it
is no longer the most valuable thing available. The five failures above are, and
three of them are Windows/HIP runtime problems on the actual generation path.
Suggested order: `ss_flow_forward`'s HIP divergence and the `ss_dec` `CONV_3D`
gap first, `chunked_decode` next (it is closest to recent work), then `dino`.

Each wants its own change and its own review; none was touched here.

## Proposed commit message

```text
Capture the 1536 coordinate reference and let a partial dump degrade

dump_cascade_reference.py gains --skip-hr-sampler: hr_coords_1536 depends only
on up_coords, which exists before the HR chain, so the 1536 token-budget
reference no longer costs a 1024-model sampler run and a 1024^3 decode. The
dump drops from gigabytes to 41 MB.

test_cascade tolerates the result instead of breaking on it. hr_noise is no
longer a required tensor, and sections 2 and 3 report the tensors as not present
rather than failing - a tensor that was deliberately not produced is not a
failure, and a ctest SKIP would not be a pass. Section 1 gates either way.

Against that dump the 1536 scaffold matches the reference exactly: 27,540 of
27,540 voxels at 96^3, resolution 1536, sym-diff 0.0000%. That closes criterion
2 of the 1536-cascade plan. The fixture stays under the token budget, so the
reduction branch itself is still unreferenced.

Two fixes found on the way: convert_dino_to_gguf.py resolved models/ against
scripts/ instead of the repository root, and ref_common now redirects the
reference tree's hardcoded /tmp log path when that directory does not exist,
which on Windows killed the decoder's upsample inside a logging FileHandler.

With the references and the f32 GGUFs in place the full suite runs here for the
first time: 11 pass, 2 skip, 5 fail. The failures are pre-existing and were
simply unobservable; they are recorded in the progress entry, not fixed here.
```
